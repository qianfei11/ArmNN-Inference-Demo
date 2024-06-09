#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Utils.hpp>
#include <armnn/Descriptors.hpp>
#include <armnn/utility/Timer.hpp>
#include <armnnOnnxParser/IOnnxParser.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
    armnn::ConfigureLogging(true, false, armnn::LogSeverity::Warning);

    // Input tensor has shape (1x1x28x28), with type of float32
    armnn::TensorShape inputTensorShape({ 1, 1, 28, 28 });

    try
    {
        // Create a runtime
        auto runtime = armnn::IRuntime::Create(armnn::IRuntime::CreationOptions());

        // Create a parser
        auto parser = armnnOnnxParser::IOnnxParser::Create();

        // Load the ONNX file into the ArmNN network
        const auto parsing_start_time = armnn::GetTimeNow();
        armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile("./mnist-12/mnist-12.onnx");
        ARMNN_LOG(info) << "Network parsing time: " << std::setprecision(2)
                        << std::fixed << armnn::GetTimeDuration(parsing_start_time).count() << " ms.";

        // Optimize the network
        const auto optimization_start_time = armnn::GetTimeNow();
        std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc, armnn::Compute::CpuAcc, armnn::Compute::CpuRef };
        armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*network, backends, runtime->GetDeviceSpec());
        ARMNN_LOG(info) << "Optimization time: " << std::setprecision(2)
                        << std::fixed << armnn::GetTimeDuration(optimization_start_time).count() << " ms.";
        if (!optNet)
        {
            throw armnn::Exception("Optimize returned nullptr");
        }

        // Load the network onto the runtime
        armnn::NetworkId networkIdentifier;
        const auto loading_start_time = armnn::GetTimeNow();
        armnn::Status ret = runtime->LoadNetwork(networkIdentifier, std::move(optNet));
        ARMNN_LOG(info) << "Network loading time: " << std::setprecision(2)
                        << std::fixed << armnn::GetTimeDuration(loading_start_time).count() << " ms.";
        if (ret == armnn::Status::Failure)
        {
            throw armnn::Exception("IRuntime::LoadNetwork failed");
        }

        // Assuming inputTensorData is your preprocessed input data
        std::vector<float> inputTensorData(10);
        armnn::InputTensors inputTensors
        {
            {0, armnn::ConstTensor(runtime->GetInputTensorInfo(networkIdentifier, 0), inputTensorData)}
        };

        // Prepare for output
        std::vector<float> outputTensorData(10); // Change this to match the output tensor shape of your model
        armnn::OutputTensors outputTensors
        {
            {0, armnn::Tensor(runtime->GetOutputTensorInfo(networkIdentifier, 0), outputTensorData.data())}
        };

        // Run inference
        const auto start_time = armnn::GetTimeNow();
        runtime->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);

        // Now outputTensorData contains the output of the model
    }
    catch (const std::exception& e)
    {
        std::cerr << "WARNING: An error has occurred: " << e.what() << std::endl;
    }

    return 0;
}
