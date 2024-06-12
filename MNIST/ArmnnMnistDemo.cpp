#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Utils.hpp>
#include <armnn/Descriptors.hpp>
#include <armnn/utility/Timer.hpp>
#include <armnnUtils/TContainer.hpp>
#include <armnnOnnxParser/IOnnxParser.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <TensorIOUtils.hpp>

// Input tensor has shape (1x1x28x28), with type of float32
armnn::TensorShape inputTensorShape({ 1, 1, 28, 28 });

std::vector<armnn::BindingPointInfo> InputBindingsInfo;
std::vector<armnn::BindingPointInfo> OutputBindingsInfo;

char modelPath[] = "./mnist-12/mnist-12.onnx";
char inputBindingName[] = "Input3";
char outputBindingName[] = "Plus214_Output_0";
char imagePath[] = "./t10k-images-idx3-ubyte";
char labelPath[] = "./t10k-labels-idx1-ubyte";

bool m_ScaleValues = true;

int g_kMnistImageByteSize = 28 * 28;

void EndianSwap(unsigned int &x)
{
    x = (x >> 24) | ((x << 8) & 0x00FF0000) | ((x >> 8) & 0x0000FF00) | (x << 24);
}

std::vector<float> GetTestCaseData(unsigned int testCaseId)
{
    std::vector<unsigned char> I(g_kMnistImageByteSize);
    unsigned int label = 0;

    std::ifstream imageStream(imagePath, std::ios::binary);
    std::ifstream labelStream(labelPath, std::ios::binary);

    if (!imageStream.is_open())
    {
        ARMNN_LOG(fatal) << "Failed to load " << imagePath;
        return std::vector<float>();
    }
    if (!labelStream.is_open())
    {
        ARMNN_LOG(fatal) << "Failed to load " << imagePath;
        return std::vector<float>();
    }

    unsigned int magic, num, row, col;

    // Checks the files have the correct header.
    imageStream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x03080000)
    {
        ARMNN_LOG(fatal) << "Failed to read " << imagePath;
        return std::vector<float>();
    }
    labelStream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x01080000)
    {
        ARMNN_LOG(fatal) << "Failed to read " << labelPath;
        return std::vector<float>();
    }

    // Endian swaps the image and label file - all the integers in the files are stored in MSB first(high endian)
    // format, hence it needs to flip the bytes of the header if using it on Intel processors or low-endian machines
    labelStream.read(reinterpret_cast<char*>(&num), sizeof(num));
    imageStream.read(reinterpret_cast<char*>(&num), sizeof(num));
    EndianSwap(num);
    imageStream.read(reinterpret_cast<char*>(&row), sizeof(row));
    EndianSwap(row);
    imageStream.read(reinterpret_cast<char*>(&col), sizeof(col));
    EndianSwap(col);

    // Reads image and label into memory.
    imageStream.seekg(testCaseId * g_kMnistImageByteSize, std::ios_base::cur);
    imageStream.read(reinterpret_cast<char*>(&I[0]), g_kMnistImageByteSize);
    labelStream.seekg(testCaseId, std::ios_base::cur);
    labelStream.read(reinterpret_cast<char*>(&label), 1);

    if (!imageStream.good())
    {
        ARMNN_LOG(fatal) << "Failed to read " << imagePath;
        return std::vector<float>();
    }
    if (!labelStream.good())
    {
        ARMNN_LOG(fatal) << "Failed to read " << labelPath;
        return std::vector<float>();
    }

    std::vector<float> inputImageData;
    inputImageData.resize(g_kMnistImageByteSize);

    for (unsigned int i = 0; i < col * row; ++i)
    {
        // Static_cast of unsigned char is safe with float
        inputImageData[i] = static_cast<float>(I[i]);

        if(m_ScaleValues)
        {
            inputImageData[i] /= 255.0f;
        }
    }

    return std::move(inputImageData);
}

armnn::InputTensors MakeInputTensors(const std::vector<armnnUtils::TContainer>& inputDataContainers)
{
    return armnnUtils::MakeInputTensors(InputBindingsInfo, inputDataContainers);
}

armnn::OutputTensors MakeOutputTensors(std::vector<armnnUtils::TContainer>& outputDataContainers)
{
    return armnnUtils::MakeOutputTensors(OutputBindingsInfo, outputDataContainers);
}

int main(int argc, char *argv[])
{
    armnn::ConfigureLogging(true, false, armnn::LogSeverity::Warning);

    try
    {
        std::vector<std::string> InputBindings = { inputBindingName };
        std::vector<std::string> OutputBindings = { outputBindingName };

        std::vector<armnn::TensorShape> InputShapesVec;
        InputShapesVec.push_back(inputTensorShape);

        std::map<std::string, armnn::TensorShape> inputShapes;

        const size_t numInputShapes = InputShapesVec.size();
        const size_t numInputBindings = InputBindings.size();
        if (numInputShapes < numInputBindings)
        {
            throw armnn::Exception(fmt::format(
                "Not every input has its tensor shape specified: expected={0}, got={1}",
                numInputBindings, numInputShapes));
        }

        for (size_t i = 0; i < numInputShapes; i++)
        {
            inputShapes[InputBindings[i]] = InputShapesVec[i];
        }

        // Create a runtime
        auto runtime = armnn::IRuntime::Create(armnn::IRuntime::CreationOptions());

        // Create a parser
        auto parser = armnnOnnxParser::IOnnxParser::Create();

        // Load the ONNX file into the ArmNN network
        const auto parsing_start_time = armnn::GetTimeNow();
        armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(modelPath);
        ARMNN_LOG(info) << "Network parsing time: " << std::setprecision(2)
                        << std::fixed << armnn::GetTimeDuration(parsing_start_time).count() << " ms.";

        for (const std::string& inputLayerName : InputBindings)
        {
            InputBindingsInfo.push_back(parser->GetNetworkInputBindingInfo(inputLayerName));
        }

        for (const std::string& outputLayerName : OutputBindings)
        {
            OutputBindingsInfo.push_back(parser->GetNetworkOutputBindingInfo(outputLayerName));
        }

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

        // Prepare for intput and output tensors
        std::vector<armnnUtils::TContainer> inputContainers = { GetTestCaseData(0) };
        std::vector<armnnUtils::TContainer> outputContainers;
        std::vector<unsigned int> outputSizes = { OutputBindingsInfo[0].second.GetNumElements() };
        const size_t numOutputs = outputSizes.size();
        outputContainers.reserve(numOutputs);
        for (size_t i = 0; i < numOutputs; i++)
        {
            outputContainers.push_back(std::vector<float>(outputSizes[i]));
        }
        armnn::InputTensors inputTensors = MakeInputTensors(inputContainers);
        armnn::OutputTensors outputTensors = MakeOutputTensors(outputContainers);

        // Run inference
        const auto start_time = armnn::GetTimeNow();
        runtime->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
        const auto duration = armnn::GetTimeDuration(start_time);

        ;
    }
    catch (const std::exception& e)
    {
        std::cerr << "WARNING: An error has occurred: " << e.what() << std::endl;
    }

    return 0;
}
