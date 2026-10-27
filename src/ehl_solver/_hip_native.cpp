#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void check_hip(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

void check_hipfft(hipfftResult status, const char* operation) {
    if (status != HIPFFT_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with hipFFT status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

template <typename T>
class DeviceAllocation {
public:
    explicit DeviceAllocation(std::size_t count) {
        check_hip(hipMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)), "hipMalloc");
    }
    ~DeviceAllocation() {
        if (pointer_ != nullptr) {
            (void)hipFree(pointer_);
        }
    }
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;
    T* get() const { return pointer_; }
    void release() {
        if (pointer_ != nullptr) {
            check_hip(hipFree(pointer_), "hipFree");
            pointer_ = nullptr;
        }
    }

private:
    T* pointer_ = nullptr;
};

class FftPlan {
public:
    FftPlan(int rows, int columns) {
        check_hipfft(hipfftPlan2d(&plan_, rows, columns, HIPFFT_Z2Z), "hipfftPlan2d");
    }
    ~FftPlan() {
        if (plan_ != 0) {
            (void)hipfftDestroy(plan_);
        }
    }
    hipfftHandle get() const { return plan_; }
    void release() {
        if (plan_ != 0) {
            check_hipfft(hipfftDestroy(plan_), "hipfftDestroy");
            plan_ = 0;
        }
    }

private:
    hipfftHandle plan_ = 0;
};

__global__ void pad_inputs(const double* signal, int signal_rows, int signal_columns,
                           const double* kernel, int kernel_rows, int kernel_columns,
                           hipfftDoubleComplex* padded_signal,
                           hipfftDoubleComplex* padded_kernel, int padded_columns,
                           std::size_t padded_size) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= padded_size) {
        return;
    }
    const int row = static_cast<int>(index / padded_columns);
    const int column = static_cast<int>(index % padded_columns);
    padded_signal[index] = make_hipDoubleComplex(
        row < signal_rows && column < signal_columns
            ? signal[static_cast<std::size_t>(row) * signal_columns + column]
            : 0.0,
        0.0);
    padded_kernel[index] = make_hipDoubleComplex(
        row < kernel_rows && column < kernel_columns
            ? kernel[static_cast<std::size_t>(row) * kernel_columns + column]
            : 0.0,
        0.0);
}

__global__ void multiply_spectra(hipfftDoubleComplex* signal,
                                const hipfftDoubleComplex* kernel,
                                std::size_t size) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }
    const double real = signal[index].x * kernel[index].x -
                        signal[index].y * kernel[index].y;
    const double imag = signal[index].x * kernel[index].y +
                        signal[index].y * kernel[index].x;
    signal[index] = make_hipDoubleComplex(real, imag);
}

__global__ void crop_result(const hipfftDoubleComplex* full, double* output, int rows,
                            int columns, int padded_columns, int crop_y, int crop_x,
                            std::size_t output_size, double normalization) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_size) {
        return;
    }
    const int row = static_cast<int>(index / columns) + crop_y;
    const int column = static_cast<int>(index % columns) + crop_x;
    output[index] =
        full[static_cast<std::size_t>(row) * padded_columns + column].x / normalization;
}

struct InputBuffer {
    Py_buffer view{};
    ~InputBuffer() {
        if (view.obj != nullptr) {
            PyBuffer_Release(&view);
        }
    }
};

bool get_double_buffer(PyObject* object, InputBuffer* buffer, const char* name) {
    if (PyObject_GetBuffer(object, &buffer->view,
                           PyBUF_FORMAT | PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) {
        return false;
    }
    if (buffer->view.ndim != 2 || buffer->view.itemsize != sizeof(double) ||
        buffer->view.format == nullptr || std::string(buffer->view.format) != "d") {
        PyErr_Format(PyExc_ValueError, "%s must be a C-contiguous 2-D float64 array", name);
        return false;
    }
    if (buffer->view.shape[0] < 1 || buffer->view.shape[1] < 1) {
        PyErr_Format(PyExc_ValueError, "%s dimensions must be positive", name);
        return false;
    }
    return true;
}

void convolve(const double* signal, int rows, int columns, const double* kernel,
              int kernel_rows, int kernel_columns, double* output, float* elapsed_ms) {
    const long long padded_rows_ll =
        static_cast<long long>(rows) + static_cast<long long>(kernel_rows) - 1;
    const long long padded_columns_ll =
        static_cast<long long>(columns) + static_cast<long long>(kernel_columns) - 1;
    if (padded_rows_ll > std::numeric_limits<int>::max() ||
        padded_columns_ll > std::numeric_limits<int>::max()) {
        throw std::overflow_error("Padded convolution dimensions exceed hipFFT limits");
    }
    const int padded_rows = static_cast<int>(padded_rows_ll);
    const int padded_columns = static_cast<int>(padded_columns_ll);
    const std::size_t padded_size =
        static_cast<std::size_t>(padded_rows) * padded_columns;
    const std::size_t output_size = static_cast<std::size_t>(rows) * columns;

    check_hip(hipSetDevice(0), "hipSetDevice");
    DeviceAllocation<double> device_signal(output_size);
    DeviceAllocation<double> device_kernel(
        static_cast<std::size_t>(kernel_rows) * kernel_columns);
    DeviceAllocation<double> device_output(output_size);
    DeviceAllocation<hipfftDoubleComplex> signal_spectrum(padded_size);
    DeviceAllocation<hipfftDoubleComplex> kernel_spectrum(padded_size);
    check_hip(hipMemcpy(device_signal.get(), signal, output_size * sizeof(double),
                        hipMemcpyHostToDevice),
              "hipMemcpy pressure to device");
    check_hip(hipMemcpy(device_kernel.get(), kernel,
                        static_cast<std::size_t>(kernel_rows) * kernel_columns *
                            sizeof(double),
                        hipMemcpyHostToDevice),
              "hipMemcpy influence kernel to device");

    FftPlan plan(padded_rows, padded_columns);
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    check_hip(hipEventCreate(&start), "hipEventCreate");
    try {
        check_hip(hipEventCreate(&stop), "hipEventCreate");
        check_hip(hipEventRecord(start), "hipEventRecord");
        constexpr unsigned int threads = 256;
        const unsigned int blocks =
            static_cast<unsigned int>((padded_size + threads - 1) / threads);
        hipLaunchKernelGGL(pad_inputs, dim3(blocks), dim3(threads), 0, 0,
                           device_signal.get(), rows, columns, device_kernel.get(),
                           kernel_rows, kernel_columns, signal_spectrum.get(),
                           kernel_spectrum.get(), padded_columns, padded_size);
        check_hip(hipGetLastError(), "pad_inputs kernel");
        check_hipfft(hipfftExecZ2Z(plan.get(), signal_spectrum.get(),
                                   signal_spectrum.get(), HIPFFT_FORWARD),
                     "hipfftExecZ2Z forward pressure");
        check_hipfft(hipfftExecZ2Z(plan.get(), kernel_spectrum.get(),
                                   kernel_spectrum.get(), HIPFFT_FORWARD),
                     "hipfftExecZ2Z forward influence kernel");
        const unsigned int product_blocks =
            static_cast<unsigned int>((padded_size + threads - 1) / threads);
        hipLaunchKernelGGL(multiply_spectra, dim3(product_blocks), dim3(threads), 0, 0,
                           signal_spectrum.get(), kernel_spectrum.get(), padded_size);
        check_hip(hipGetLastError(), "multiply_spectra kernel");
        check_hipfft(hipfftExecZ2Z(plan.get(), signal_spectrum.get(),
                                   signal_spectrum.get(), HIPFFT_BACKWARD),
                     "hipfftExecZ2Z inverse");
        const unsigned int crop_blocks =
            static_cast<unsigned int>((output_size + threads - 1) / threads);
        hipLaunchKernelGGL(crop_result, dim3(crop_blocks), dim3(threads), 0, 0,
                           signal_spectrum.get(), device_output.get(), rows, columns,
                           padded_columns, kernel_rows / 2, kernel_columns / 2,
                           output_size, static_cast<double>(padded_size));
        check_hip(hipGetLastError(), "crop_result kernel");
        check_hip(hipEventRecord(stop), "hipEventRecord");
        check_hip(hipEventSynchronize(stop), "hipEventSynchronize");
        if (elapsed_ms != nullptr) {
            check_hip(hipEventElapsedTime(elapsed_ms, start, stop),
                      "hipEventElapsedTime");
        }
        check_hip(hipMemcpy(output, device_output.get(), output_size * sizeof(double),
                            hipMemcpyDeviceToHost),
                  "hipMemcpy result to host");
    } catch (...) {
        if (stop != nullptr) {
            (void)hipEventDestroy(stop);
        }
        (void)hipEventDestroy(start);
        throw;
    }
    if (stop != nullptr) {
        check_hip(hipEventDestroy(stop), "hipEventDestroy");
    }
    check_hip(hipEventDestroy(start), "hipEventDestroy");
    plan.release();
    device_output.release();
    kernel_spectrum.release();
    signal_spectrum.release();
    device_kernel.release();
    device_signal.release();
}

PyObject* create_output(int rows, int columns) {
    PyObject* numpy = PyImport_ImportModule("numpy");
    if (numpy == nullptr) {
        return nullptr;
    }
    PyObject* shape = Py_BuildValue("(ii)", rows, columns);
    PyObject* output =
        shape == nullptr ? nullptr : PyObject_CallMethod(numpy, "empty", "Os", shape, "float64");
    Py_XDECREF(shape);
    Py_DECREF(numpy);
    return output;
}

PyObject* run_convolution(PyObject* signal_object, PyObject* kernel_object, bool timed) {
    InputBuffer signal_buffer;
    InputBuffer kernel_buffer;
    if (!get_double_buffer(signal_object, &signal_buffer, "signal") ||
        !get_double_buffer(kernel_object, &kernel_buffer, "kernel")) {
        return nullptr;
    }
    const int rows = static_cast<int>(signal_buffer.view.shape[0]);
    const int columns = static_cast<int>(signal_buffer.view.shape[1]);
    const int kernel_rows = static_cast<int>(kernel_buffer.view.shape[0]);
    const int kernel_columns = static_cast<int>(kernel_buffer.view.shape[1]);
    PyObject* output_object = create_output(rows, columns);
    if (output_object == nullptr) {
        return nullptr;
    }
    Py_buffer output_buffer{};
    if (PyObject_GetBuffer(output_object, &output_buffer,
                           PyBUF_WRITABLE | PyBUF_FORMAT | PyBUF_ND |
                               PyBUF_C_CONTIGUOUS) < 0) {
        Py_DECREF(output_object);
        return nullptr;
    }

    float elapsed_ms = 0.0F;
    std::string error;
    PyThreadState* thread_state = PyEval_SaveThread();
    try {
        convolve(static_cast<const double*>(signal_buffer.view.buf), rows, columns,
                 static_cast<const double*>(kernel_buffer.view.buf), kernel_rows,
                 kernel_columns, static_cast<double*>(output_buffer.buf),
                 timed ? &elapsed_ms : nullptr);
    } catch (const std::exception& exception) {
        error = exception.what();
    }
    PyEval_RestoreThread(thread_state);
    PyBuffer_Release(&output_buffer);
    if (!error.empty()) {
        Py_DECREF(output_object);
        PyErr_SetString(PyExc_RuntimeError, error.c_str());
        return nullptr;
    }
    if (!timed) {
        return output_object;
    }
    PyObject* result = Py_BuildValue("(Nf)", output_object, elapsed_ms);
    return result;
}

PyObject* device_count(PyObject*, PyObject*) {
    int count = 0;
    const hipError_t status = hipGetDeviceCount(&count);
    if (status != hipSuccess) {
        PyErr_Format(PyExc_RuntimeError, "hipGetDeviceCount: %s", hipGetErrorString(status));
        return nullptr;
    }
    return PyLong_FromLong(count);
}

PyObject* device_info(PyObject*, PyObject*) {
    const hipError_t selected = hipSetDevice(0);
    if (selected != hipSuccess) {
        PyErr_Format(PyExc_RuntimeError, "hipSetDevice: %s", hipGetErrorString(selected));
        return nullptr;
    }
    hipDeviceProp_t properties{};
    hipError_t status = hipGetDeviceProperties(&properties, 0);
    if (status != hipSuccess) {
        PyErr_Format(PyExc_RuntimeError, "hipGetDeviceProperties: %s", hipGetErrorString(status));
        return nullptr;
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    status = hipMemGetInfo(&free_bytes, &total_bytes);
    if (status != hipSuccess) {
        PyErr_Format(PyExc_RuntimeError, "hipMemGetInfo: %s", hipGetErrorString(status));
        return nullptr;
    }
    int runtime_version = 0;
    status = hipRuntimeGetVersion(&runtime_version);
    if (status != hipSuccess) {
        PyErr_Format(PyExc_RuntimeError, "hipRuntimeGetVersion: %s", hipGetErrorString(status));
        return nullptr;
    }
    int fft_version = 0;
    const hipfftResult fft_status = hipfftGetVersion(&fft_version);
    if (fft_status != HIPFFT_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, "hipfftGetVersion failed");
        return nullptr;
    }
    return Py_BuildValue("{s:s,s:s,s:K,s:K,s:i,s:i}",
                         "name", properties.name,
                         "architecture", properties.gcnArchName,
                         "free_memory_bytes", static_cast<unsigned long long>(free_bytes),
                         "total_memory_bytes", static_cast<unsigned long long>(total_bytes),
                         "runtime_version", runtime_version,
                         "hipfft_version", fft_version);
}

PyObject* convolve_same(PyObject*, PyObject* args) {
    PyObject* signal = nullptr;
    PyObject* kernel = nullptr;
    if (!PyArg_ParseTuple(args, "OO:convolve_same", &signal, &kernel)) {
        return nullptr;
    }
    return run_convolution(signal, kernel, false);
}

PyObject* convolve_same_timed(PyObject*, PyObject* args) {
    PyObject* signal = nullptr;
    PyObject* kernel = nullptr;
    if (!PyArg_ParseTuple(args, "OO:convolve_same_timed", &signal, &kernel)) {
        return nullptr;
    }
    return run_convolution(signal, kernel, true);
}

PyMethodDef methods[] = {
    {"device_count", device_count, METH_NOARGS,
     "Return the number of HIP devices or raise if the runtime fails."},
    {"device_info", device_info, METH_NOARGS,
     "Return the active device architecture, memory availability, and library versions."},
    {"convolve_same", convolve_same, METH_VARARGS,
     "Compute a zero-padded float64 same-mode 2-D convolution."},
    {"convolve_same_timed", convolve_same_timed, METH_VARARGS,
     "Compute convolution and return (result, device execution milliseconds)."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_hip_native", nullptr, -1, methods};

}  // namespace

PyMODINIT_FUNC PyInit__hip_native() {
    return PyModule_Create(&module);
}
