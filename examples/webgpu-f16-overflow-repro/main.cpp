#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr uint64_t kWaitTimeoutNs = 60ull * 1000ull * 1000ull * 1000ull;

template <typename T>
T bit_cast_copy(const void * ptr) {
    T value;
    std::memcpy(&value, ptr, sizeof(T));
    return value;
}

static float decode_fp16(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x03ffu;

    uint32_t f = 0;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            uint32_t mantissa = mant;
            uint32_t exponent = 127u - 15u + 1u;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffu;
            f = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exp == 0x1fu) {
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        const uint32_t exponent = exp + (127u - 15u);
        f = sign | (exponent << 23) | (mant << 13);
    }

    return bit_cast_copy<float>(&f);
}

static std::string adapter_type_name(wgpu::AdapterType type) {
    switch (type) {
        case wgpu::AdapterType::DiscreteGPU: return "DiscreteGPU";
        case wgpu::AdapterType::IntegratedGPU: return "IntegratedGPU";
        case wgpu::AdapterType::CPU: return "CPU";
        case wgpu::AdapterType::Unknown: return "Unknown";
        default: return "Other";
    }
}

static std::string backend_name(wgpu::BackendType type) {
    switch (type) {
        case wgpu::BackendType::Undefined: return "Undefined";
        case wgpu::BackendType::Null: return "Null";
        case wgpu::BackendType::WebGPU: return "WebGPU";
        case wgpu::BackendType::D3D11: return "D3D11";
        case wgpu::BackendType::D3D12: return "D3D12";
        case wgpu::BackendType::Metal: return "Metal";
        case wgpu::BackendType::Vulkan: return "Vulkan";
        case wgpu::BackendType::OpenGL: return "OpenGL";
        case wgpu::BackendType::OpenGLES: return "OpenGLES";
        default: return "Other";
    }
}

static void check_wait_status(wgpu::WaitStatus status, const char * what) {
    if (status == wgpu::WaitStatus::Success) {
        return;
    }

    std::cerr << what << " failed with WaitStatus=" << static_cast<int>(status) << '\n';
    std::exit(1);
}

struct WebGPUContext {
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;
};

static std::optional<wgpu::Adapter> request_adapter(
    const wgpu::Instance & instance,
    wgpu::BackendType backend_type,
    std::string & error_out) {
    wgpu::Adapter adapter;
    wgpu::RequestAdapterOptions adapter_options = {};
    adapter_options.backendType = backend_type;

#ifndef __EMSCRIPTEN__
    const char * const adapter_enabled_toggles[] = {
        "vulkan_enable_f16_on_nvidia",
        "use_vulkan_memory_model",
    };
    wgpu::DawnTogglesDescriptor adapter_toggles = {};
    adapter_toggles.enabledToggles = adapter_enabled_toggles;
    adapter_toggles.enabledToggleCount = 2;
    adapter_options.nextInChain = &adapter_toggles;
#endif

    check_wait_status(
        instance.WaitAny(
            instance.RequestAdapter(
                &adapter_options,
                wgpu::CallbackMode::AllowSpontaneous,
                [&adapter, &error_out](wgpu::RequestAdapterStatus status, wgpu::Adapter value, const char * message) {
                    if (status != wgpu::RequestAdapterStatus::Success) {
                        error_out = message ? message : "";
                        return;
                    }
                    adapter = std::move(value);
                }),
            kWaitTimeoutNs),
        "request adapter");

    if (adapter == nullptr) {
        return std::nullopt;
    }

    wgpu::AdapterInfo info = {};
    adapter.GetInfo(&info);
    if (info.backendType == wgpu::BackendType::Null) {
        error_out = "adapter resolved to Null backend";
        return std::nullopt;
    }

    return adapter;
}

static WebGPUContext init_webgpu() {
    WebGPUContext ctx;

    wgpu::InstanceDescriptor instance_desc = {};
    const std::array instance_features = { wgpu::InstanceFeatureName::TimedWaitAny };
    instance_desc.requiredFeatures = instance_features.data();
    instance_desc.requiredFeatureCount = instance_features.size();

#ifndef __EMSCRIPTEN__
    const char * const instance_enabled_toggles[] = { "allow_unsafe_apis" };
    wgpu::DawnTogglesDescriptor instance_toggles = {};
    instance_toggles.enabledToggles = instance_enabled_toggles;
    instance_toggles.enabledToggleCount = 1;
    instance_desc.nextInChain = &instance_toggles;
#endif

    ctx.instance = wgpu::CreateInstance(&instance_desc);
    if (ctx.instance == nullptr) {
        std::cerr << "failed to create WebGPU instance\n";
        std::exit(1);
    }

    std::string adapter_error;

#if defined(__APPLE__)
    const std::array backend_preference = {
        wgpu::BackendType::Metal,
        wgpu::BackendType::Vulkan,
        wgpu::BackendType::OpenGL,
    };
#elif defined(_WIN32)
    const std::array backend_preference = {
        wgpu::BackendType::D3D12,
        wgpu::BackendType::Vulkan,
        wgpu::BackendType::OpenGL,
    };
#else
    const std::array backend_preference = {
        wgpu::BackendType::Vulkan,
        wgpu::BackendType::Metal,
        wgpu::BackendType::OpenGL,
    };
#endif

    for (wgpu::BackendType backend_type : backend_preference) {
        std::string attempt_error;
        std::optional<wgpu::Adapter> adapter = request_adapter(ctx.instance, backend_type, attempt_error);
        if (adapter.has_value()) {
            ctx.adapter = std::move(*adapter);
            break;
        }
        adapter_error = attempt_error;
    }

    if (ctx.adapter == nullptr) {
        std::cerr << "failed to get a non-null adapter";
        if (!adapter_error.empty()) {
            std::cerr << ": " << adapter_error;
        }
        std::cerr << '\n';
        std::exit(1);
    }

    if (!ctx.adapter.HasFeature(wgpu::FeatureName::ShaderF16)) {
        std::cerr << "adapter does not support ShaderF16\n";
        std::exit(1);
    }

    wgpu::AdapterInfo info = {};
    ctx.adapter.GetInfo(&info);

    std::cout << "adapter: " << std::string(info.description) << '\n';
    std::cout << "backend: " << backend_name(info.backendType) << '\n';
    std::cout << "adapter_type: " << adapter_type_name(info.adapterType) << '\n';

    std::vector<wgpu::FeatureName> required_features = { wgpu::FeatureName::ShaderF16 };
#ifndef __EMSCRIPTEN__
    required_features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
#endif

    wgpu::DeviceDescriptor device_desc = {};
    device_desc.requiredFeatures = required_features.data();
    device_desc.requiredFeatureCount = required_features.size();
    device_desc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device &, wgpu::DeviceLostReason reason, wgpu::StringView message) {
            if (reason == wgpu::DeviceLostReason::Destroyed) {
                return;
            }
            std::cerr << "device lost: reason=" << static_cast<int>(reason)
                      << " message=" << std::string(message) << '\n';
            std::exit(1);
        });
    device_desc.SetUncapturedErrorCallback(
        [](const wgpu::Device &, wgpu::ErrorType type, wgpu::StringView message) {
            std::cerr << "uncaptured device error: type=" << static_cast<int>(type)
                      << " message=" << std::string(message) << '\n';
            std::exit(1);
        });

    std::string device_error;
    check_wait_status(
        ctx.instance.WaitAny(
            ctx.adapter.RequestDevice(
                &device_desc,
                wgpu::CallbackMode::AllowSpontaneous,
                [&ctx, &device_error](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
                    if (status != wgpu::RequestDeviceStatus::Success) {
                        device_error = std::string(message);
                        return;
                    }
                    ctx.device = std::move(device);
                }),
            kWaitTimeoutNs),
        "request device");

    if (ctx.device == nullptr) {
        std::cerr << "failed to get device";
        if (!device_error.empty()) {
            std::cerr << ": " << device_error;
        }
        std::cerr << '\n';
        std::exit(1);
    }

    ctx.queue = ctx.device.GetQueue();
    return ctx;
}

static wgpu::Buffer create_buffer(
    const wgpu::Device & device,
    uint64_t size,
    wgpu::BufferUsage usage,
    const char * label,
    const void * initial_data = nullptr) {
    wgpu::BufferDescriptor desc = {};
    desc.label = label;
    desc.size = size;
    desc.usage = usage;
    desc.mappedAtCreation = initial_data != nullptr;

    wgpu::Buffer buffer = device.CreateBuffer(&desc);
    if (buffer == nullptr) {
        std::cerr << "failed to create buffer: " << label << '\n';
        std::exit(1);
    }

    if (initial_data != nullptr) {
        std::memcpy(buffer.GetMappedRange(), initial_data, static_cast<size_t>(size));
        buffer.Unmap();
    }

    return buffer;
}

static std::vector<uint16_t> map_read_u16(WebGPUContext & ctx, wgpu::Buffer & buffer, size_t count) {
    std::vector<uint16_t> out(count);
    wgpu::MapAsyncStatus map_status = wgpu::MapAsyncStatus::Error;
    std::string map_message;

    check_wait_status(
        ctx.instance.WaitAny(
            buffer.MapAsync(
                wgpu::MapMode::Read,
                0,
                count * sizeof(uint16_t),
                wgpu::CallbackMode::AllowSpontaneous,
                [&map_status, &map_message](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                    map_status = status;
                    map_message = std::string(message);
                }),
            kWaitTimeoutNs),
        "map buffer");

    if (map_status != wgpu::MapAsyncStatus::Success) {
        std::cerr << "failed to map read buffer: " << map_message << '\n';
        std::exit(1);
    }

    const void * mapped = buffer.GetConstMappedRange();
    std::memcpy(out.data(), mapped, count * sizeof(uint16_t));
    buffer.Unmap();
    return out;
}

static std::vector<uint32_t> map_read_u32(WebGPUContext & ctx, wgpu::Buffer & buffer, size_t count) {
    std::vector<uint32_t> out(count);
    wgpu::MapAsyncStatus map_status = wgpu::MapAsyncStatus::Error;
    std::string map_message;

    check_wait_status(
        ctx.instance.WaitAny(
            buffer.MapAsync(
                wgpu::MapMode::Read,
                0,
                count * sizeof(uint32_t),
                wgpu::CallbackMode::AllowSpontaneous,
                [&map_status, &map_message](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                    map_status = status;
                    map_message = std::string(message);
                }),
            kWaitTimeoutNs),
        "map buffer");

    if (map_status != wgpu::MapAsyncStatus::Success) {
        std::cerr << "failed to map read buffer: " << map_message << '\n';
        std::exit(1);
    }

    const void * mapped = buffer.GetConstMappedRange();
    std::memcpy(out.data(), mapped, count * sizeof(uint32_t));
    buffer.Unmap();
    return out;
}

}  // namespace

int main() {
    const std::vector<float> inputs = {
        10.5f,
        11.0f,
        11.0859375f,
        11.09375f,
        11.125f,
        11.5f,
        12.0f,
        13.0f,
    };
    const uint32_t count = static_cast<uint32_t>(inputs.size());

    std::ostringstream wgsl;
    wgsl
        << "enable f16;\n"
        << "struct InputBuf {\n"
        << "  values: array<f32, " << count << ">,\n"
        << "};\n"
        << "struct OutputF16Buf {\n"
        << "  values: array<f16, " << count << ">,\n"
        << "};\n"
        << "struct OutputBuf {\n"
        << "  values: array<u32, " << count << ">,\n"
        << "};\n"
        << "@group(0) @binding(0) var<storage, read> in_buf: InputBuf;\n"
        << "@group(0) @binding(1) var<storage, read_write> out_exp: OutputF16Buf;\n"
        << "@group(0) @binding(2) var<storage, read_write> out_expm1: OutputF16Buf;\n"
        << "@group(0) @binding(3) var<storage, read_write> out_exp_f32_bits: OutputBuf;\n"
        << "@group(0) @binding(4) var<storage, read_write> out_expm1_f32_bits: OutputBuf;\n"
        << "@compute @workgroup_size(64)\n"
        << "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        << "  let i = gid.x;\n"
        << "  if (i >= " << count << "u) {\n"
        << "    return;\n"
        << "  }\n"
        << "  let x = in_buf.values[i];\n"
        << "  let exp_f32 = exp(x);\n"
        << "  let expm1_f32 = exp(x) - 1.0;\n"
        << "  let exp_f16 = f16(exp_f32);\n"
        << "  let expm1_f16 = f16(expm1_f32);\n"
        << "  out_exp.values[i] = exp_f16;\n"
        << "  out_expm1.values[i] = expm1_f16;\n"
        << "  out_exp_f32_bits.values[i] = bitcast<u32>(exp_f32);\n"
        << "  out_expm1_f32_bits.values[i] = bitcast<u32>(expm1_f32);\n"
        << "}\n";

    WebGPUContext ctx = init_webgpu();

    wgpu::ShaderSourceWGSL shader_source = {};
    const std::string shader_text = wgsl.str();
    shader_source.code = shader_text.c_str();

    wgpu::ShaderModuleDescriptor shader_desc = {};
    shader_desc.nextInChain = &shader_source;
    wgpu::ShaderModule shader = ctx.device.CreateShaderModule(&shader_desc);

    wgpu::ComputePipelineDescriptor pipeline_desc = {};
    pipeline_desc.compute.module = shader;
    pipeline_desc.compute.entryPoint = "main";
    wgpu::ComputePipeline pipeline = ctx.device.CreateComputePipeline(&pipeline_desc);

    const uint64_t input_bytes = inputs.size() * sizeof(float);
    const uint64_t output_f16_bytes = inputs.size() * sizeof(uint16_t);
    const uint64_t output_u32_bytes = inputs.size() * sizeof(uint32_t);

    wgpu::Buffer input_buf = create_buffer(
        ctx.device,
        input_bytes,
        wgpu::BufferUsage::Storage,
        "input_buf",
        inputs.data());
    wgpu::Buffer exp_buf = create_buffer(
        ctx.device,
        output_f16_bytes,
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
        "exp_buf");
    wgpu::Buffer expm1_buf = create_buffer(
        ctx.device,
        output_f16_bytes,
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
        "expm1_buf");
    wgpu::Buffer exp_f32_bits_buf = create_buffer(
        ctx.device,
        output_u32_bytes,
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
        "exp_f32_bits_buf");
    wgpu::Buffer expm1_f32_bits_buf = create_buffer(
        ctx.device,
        output_u32_bytes,
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
        "expm1_f32_bits_buf");

    wgpu::Buffer exp_readback = create_buffer(
        ctx.device,
        output_f16_bytes,
        wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
        "exp_readback");
    wgpu::Buffer expm1_readback = create_buffer(
        ctx.device,
        output_f16_bytes,
        wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
        "expm1_readback");
    wgpu::Buffer exp_f32_bits_readback = create_buffer(
        ctx.device,
        output_u32_bytes,
        wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
        "exp_f32_bits_readback");
    wgpu::Buffer expm1_f32_bits_readback = create_buffer(
        ctx.device,
        output_u32_bytes,
        wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
        "expm1_f32_bits_readback");

    std::array<wgpu::BindGroupEntry, 5> entries = {};
    entries[0].binding = 0;
    entries[0].buffer = input_buf;
    entries[0].size = input_bytes;
    entries[1].binding = 1;
    entries[1].buffer = exp_buf;
    entries[1].size = output_f16_bytes;
    entries[2].binding = 2;
    entries[2].buffer = expm1_buf;
    entries[2].size = output_f16_bytes;
    entries[3].binding = 3;
    entries[3].buffer = exp_f32_bits_buf;
    entries[3].size = output_u32_bytes;
    entries[4].binding = 4;
    entries[4].buffer = expm1_f32_bits_buf;
    entries[4].size = output_u32_bytes;

    wgpu::BindGroupDescriptor bind_group_desc = {};
    bind_group_desc.layout = pipeline.GetBindGroupLayout(0);
    bind_group_desc.entryCount = entries.size();
    bind_group_desc.entries = entries.data();
    wgpu::BindGroup bind_group = ctx.device.CreateBindGroup(&bind_group_desc);

    wgpu::CommandEncoder encoder = ctx.device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(1);
    pass.End();

    encoder.CopyBufferToBuffer(exp_buf, 0, exp_readback, 0, output_f16_bytes);
    encoder.CopyBufferToBuffer(expm1_buf, 0, expm1_readback, 0, output_f16_bytes);
    encoder.CopyBufferToBuffer(exp_f32_bits_buf, 0, exp_f32_bits_readback, 0, output_u32_bytes);
    encoder.CopyBufferToBuffer(expm1_f32_bits_buf, 0, expm1_f32_bits_readback, 0, output_u32_bytes);

    wgpu::CommandBuffer commands = encoder.Finish();
    ctx.queue.Submit(1, &commands);

    wgpu::QueueWorkDoneStatus done_status = wgpu::QueueWorkDoneStatus::Error;
    std::string done_message;
    check_wait_status(
        ctx.instance.WaitAny(
            ctx.queue.OnSubmittedWorkDone(
                wgpu::CallbackMode::AllowSpontaneous,
                [&done_status, &done_message](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
                    done_status = status;
                    done_message = std::string(message);
                }),
            kWaitTimeoutNs),
        "queue work done");

    if (done_status != wgpu::QueueWorkDoneStatus::Success) {
        std::cerr << "queue completion failed: " << done_message << '\n';
        return 1;
    }

    const std::vector<uint16_t> exp_half_bits = map_read_u16(ctx, exp_readback, inputs.size());
    const std::vector<uint16_t> expm1_half_bits = map_read_u16(ctx, expm1_readback, inputs.size());
    const std::vector<uint32_t> exp_f32_bits = map_read_u32(ctx, exp_f32_bits_readback, inputs.size());
    const std::vector<uint32_t> expm1_f32_bits = map_read_u32(ctx, expm1_f32_bits_readback, inputs.size());

    std::cout << std::fixed << std::setprecision(8);
    std::cout << '\n';

    for (size_t i = 0; i < inputs.size(); ++i) {
        const float x = inputs[i];
        const float cpu_exp = std::exp(x);
        const float cpu_expm1 = std::exp(x) - 1.0f;

        const float gpu_exp_f32 = bit_cast_copy<float>(&exp_f32_bits[i]);
        const float gpu_expm1_f32 = bit_cast_copy<float>(&expm1_f32_bits[i]);

        std::cout << "x=" << x << '\n';
        std::cout << "  exp_f32_cpu=" << cpu_exp << " exp_f32_gpu=" << gpu_exp_f32 << '\n';
        std::cout << "  exp_f16_bits=0x" << std::hex << std::setw(4) << std::setfill('0') << exp_half_bits[i]
                  << std::dec << std::setfill(' ') << " exp_f16=" << decode_fp16(exp_half_bits[i]) << '\n';
        std::cout << "  expm1_f32_cpu=" << cpu_expm1 << " expm1_f32_gpu=" << gpu_expm1_f32 << '\n';
        std::cout << "  expm1_f16_bits=0x" << std::hex << std::setw(4) << std::setfill('0') << expm1_half_bits[i]
                  << std::dec << std::setfill(' ') << " expm1_f16=" << decode_fp16(expm1_half_bits[i]) << '\n';
        std::cout << '\n';
    }

    return 0;
}
