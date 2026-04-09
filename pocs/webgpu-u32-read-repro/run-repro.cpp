#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kOutFloats = 5;
constexpr uint32_t kDebugWords = 4;
constexpr uint32_t kExpected[kDebugWords] = {
    0xdeadbeefu,
    1u,
    2u,
    3u,
};
constexpr uint32_t kInputWords[3] = {
    0x04030201u,
    0x08070605u,
    0x0c0b0a09u,
};

struct Options {
    std::string shader_path = "pocs/webgpu-u32-read-repro/mul_mat_reg_tile_q4_K_f32_exact.wgsl";
    bool disable_robustness = false;
    bool disable_polyfill_divmod = false;
};

std::string read_text_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open shader: " + path);
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string to_string(wgpu::StringView s) {
    return std::string(s);
}

Options parse_args(int argc, char ** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--disable-robustness") {
            opts.disable_robustness = true;
        } else if (arg == "--disable-polyfill-divmod") {
            opts.disable_polyfill_divmod = true;
        } else if (arg == "--shader" && i + 1 < argc) {
            opts.shader_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: webgpu-u32-read-repro-runner [--shader PATH] "
                   "[--disable-robustness] [--disable-polyfill-divmod]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return opts;
}

template <typename TStatus>
void check_wait_status(wgpu::WaitStatus wait_status,
                       TStatus callback_status,
                       TStatus expected_status,
                       const std::string & what,
                       const std::string & message) {
    if (wait_status != wgpu::WaitStatus::Success) {
        throw std::runtime_error(what + " wait failed");
    }
    if (callback_status != expected_status) {
        throw std::runtime_error(what + " failed: " + message);
    }
}

wgpu::Adapter request_adapter(wgpu::Instance & instance) {
    wgpu::RequestAdapterOptions options = {};
    wgpu::Adapter adapter;
    wgpu::RequestAdapterStatus status = wgpu::RequestAdapterStatus::Unavailable;
    std::string message;

    auto future = instance.RequestAdapter(
        &options, wgpu::CallbackMode::AllowSpontaneous,
        [&](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView m) {
            status = s;
            adapter = std::move(a);
            message = std::string(m);
        });

    auto wait_status = instance.WaitAny(future, std::numeric_limits<uint64_t>::max());
    check_wait_status(wait_status, status, wgpu::RequestAdapterStatus::Success, "request adapter", message);
    return adapter;
}

wgpu::Device request_device(wgpu::Instance & instance, wgpu::Adapter & adapter, const Options & opts) {
    wgpu::DeviceDescriptor desc = {};
    desc.SetUncapturedErrorCallback([](const wgpu::Device &, wgpu::ErrorType type, wgpu::StringView message) {
        std::cerr << "uncaptured error: " << static_cast<int>(type) << ": " << std::string(message) << "\n";
    });

    std::vector<const char *> enabled_toggles;
    if (opts.disable_robustness) {
        enabled_toggles.push_back("disable_robustness");
    }
    if (opts.disable_polyfill_divmod) {
        enabled_toggles.push_back("disable_polyfills_on_integer_div_and_mod");
    }

    wgpu::DawnTogglesDescriptor toggles = {};
    if (!enabled_toggles.empty()) {
        toggles.enabledToggles = enabled_toggles.data();
        toggles.enabledToggleCount = enabled_toggles.size();
        desc.nextInChain = &toggles;
    }

    wgpu::Device device;
    wgpu::RequestDeviceStatus status = wgpu::RequestDeviceStatus::Error;
    std::string message;

    auto future = adapter.RequestDevice(
        &desc, wgpu::CallbackMode::AllowSpontaneous,
        [&](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView m) {
            status = s;
            device = std::move(d);
            message = std::string(m);
        });

    auto wait_status = instance.WaitAny(future, std::numeric_limits<uint64_t>::max());
    check_wait_status(wait_status, status, wgpu::RequestDeviceStatus::Success, "request device", message);
    return device;
}

void map_read_buffer(wgpu::Instance & instance, wgpu::Buffer & buffer, size_t size) {
    wgpu::MapAsyncStatus status = wgpu::MapAsyncStatus::Error;
    std::string message;
    auto future = buffer.MapAsync(
        wgpu::MapMode::Read, 0, size, wgpu::CallbackMode::AllowSpontaneous,
        [&](wgpu::MapAsyncStatus s, wgpu::StringView m) {
            status = s;
            message = std::string(m);
        });
    auto wait_status = instance.WaitAny(future, std::numeric_limits<uint64_t>::max());
    check_wait_status(wait_status, status, wgpu::MapAsyncStatus::Success, "map buffer", message);
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const Options opts = parse_args(argc, argv);
        const std::string shader = read_text_file(opts.shader_path);

        static constexpr auto kTimedWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
        wgpu::InstanceDescriptor instance_desc = {};
        instance_desc.requiredFeatures = &kTimedWaitAny;
        instance_desc.requiredFeatureCount = 1;

        wgpu::Instance instance = wgpu::CreateInstance(&instance_desc);
        if (instance == nullptr) {
            throw std::runtime_error("failed to create instance");
        }

        wgpu::Adapter adapter = request_adapter(instance);
        wgpu::AdapterInfo info = {};
        adapter.GetInfo(&info);

        wgpu::Device device = request_device(instance, adapter, opts);
        wgpu::Queue queue = device.GetQueue();

        wgpu::ShaderSourceWGSL shader_source = {};
        shader_source.code = shader.c_str();
        wgpu::ShaderModuleDescriptor shader_desc = {};
        shader_desc.nextInChain = &shader_source;
        wgpu::ShaderModule module = device.CreateShaderModule(&shader_desc);

        wgpu::ComputePipelineDescriptor pipeline_desc = {};
        pipeline_desc.compute.module = module;
        pipeline_desc.compute.entryPoint = "main";
        pipeline_desc.layout = nullptr;
        wgpu::ComputePipeline pipeline = device.CreateComputePipeline(&pipeline_desc);

        wgpu::BufferDescriptor src_desc = {};
        src_desc.size = sizeof(kInputWords);
        src_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        wgpu::Buffer src = device.CreateBuffer(&src_desc);
        queue.WriteBuffer(src, 0, kInputWords, sizeof(kInputWords));

        wgpu::BufferDescriptor dst_desc = {};
        dst_desc.size = kOutFloats * sizeof(float);
        dst_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
        wgpu::Buffer dst = device.CreateBuffer(&dst_desc);

        wgpu::BufferDescriptor debug_desc = {};
        debug_desc.size = kDebugWords * sizeof(uint32_t);
        debug_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
        wgpu::Buffer debug = device.CreateBuffer(&debug_desc);

        wgpu::BufferDescriptor read_desc = {};
        read_desc.size = kDebugWords * sizeof(uint32_t);
        read_desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
        wgpu::Buffer debug_read = device.CreateBuffer(&read_desc);

        wgpu::BindGroup bind_group;
        {
            wgpu::BindGroupEntry entries[3] = {};
            entries[0].binding = 0;
            entries[0].buffer = src;
            entries[0].offset = 0;
            entries[0].size = sizeof(kInputWords);
            entries[1].binding = 1;
            entries[1].buffer = dst;
            entries[1].offset = 0;
            entries[1].size = kOutFloats * sizeof(float);
            entries[2].binding = 2;
            entries[2].buffer = debug;
            entries[2].offset = 0;
            entries[2].size = kDebugWords * sizeof(uint32_t);

            wgpu::BindGroupDescriptor bg_desc = {};
            bg_desc.layout = pipeline.GetBindGroupLayout(0);
            bg_desc.entryCount = 3;
            bg_desc.entries = entries;
            bind_group = device.CreateBindGroup(&bg_desc);
        }

        wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bind_group);
        pass.DispatchWorkgroups(1, 1, 1);
        pass.End();
        encoder.CopyBufferToBuffer(debug, 0, debug_read, 0, kDebugWords * sizeof(uint32_t));
        wgpu::CommandBuffer commands = encoder.Finish();
        queue.Submit(1, &commands);

        {
            wgpu::QueueWorkDoneStatus status = wgpu::QueueWorkDoneStatus::Error;
            std::string message;
            auto future = queue.OnSubmittedWorkDone(
                wgpu::CallbackMode::AllowSpontaneous,
                [&](wgpu::QueueWorkDoneStatus s, wgpu::StringView m) {
                    status = s;
                    message = std::string(m);
                });
            auto wait_status = instance.WaitAny(future, std::numeric_limits<uint64_t>::max());
            check_wait_status(wait_status, status, wgpu::QueueWorkDoneStatus::Success, "queue submit", message);
        }

        map_read_buffer(instance, debug_read, kDebugWords * sizeof(uint32_t));
        const auto * mapped = static_cast<const uint32_t *>(debug_read.GetConstMappedRange());

        std::cout << "adapter: " << to_string(info.vendor) << " / " << to_string(info.architecture) << " / "
                  << to_string(info.description) << "\n";
        std::cout << "shader: " << opts.shader_path << "\n";
        std::cout << "toggles:\n";
        std::cout << "  disable_robustness: " << (opts.disable_robustness ? "yes" : "no") << "\n";
        std::cout << "  disable_polyfills_on_integer_div_and_mod: " << (opts.disable_polyfill_divmod ? "yes" : "no")
                  << "\n";
        std::cout << "input:\n";
        for (size_t i = 0; i < std::size(kInputWords); ++i) {
            std::cout << "  src0[" << i << "] = 0x" << std::hex << kInputWords[i] << std::dec << " (" << kInputWords[i]
                      << ")\n";
        }
        std::cout << "debug:\n";
        for (uint32_t i = 0; i < kDebugWords; ++i) {
            const bool ok = mapped[i] == kExpected[i];
            std::cout << "  dbg[" << i << "] = 0x" << std::hex << mapped[i] << std::dec << " (" << mapped[i]
                      << ") expected 0x" << std::hex << kExpected[i] << std::dec << " (" << kExpected[i] << ") ["
                      << (ok ? "ok" : "mismatch") << "]\n";
        }

        debug_read.Unmap();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
