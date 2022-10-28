#include <iterator>
#include <vector>
#include <memory>
#include <string>
#include <vector>
#include <wrl.h>
#include <chrono>
#include <Windows.h>
#include <filesystem>

#undef max
#undef min
#include <ie/inference_engine.hpp>

#define max(a, b) (a > b ? a : b)
#define min(a, b) (a < b ? a : b)

using namespace InferenceEngine;

// declare function 
void UserInput();
void PrintConfig();
bool CheckExist(const std::string src);

// global variable
std::string model_path;
std::string bin_path;

int input_width;
int input_height;
int inference_times;
int model_id;
float sleep;

std::string infer_device;
std::string user_input_str;

// entry points
int main(int argc, char* argv[]) {

    UserInput();

    // check model exsit or not
    if (!CheckExist(model_path) || !CheckExist(bin_path)) {
        printf("Model not exsit (check both .xml and .bin file in root folder) ...");
        Sleep(10000);
        exit;
    }

    // print configuration
    PrintConfig();

    try {
        // step 1. initialize inference engine core
        Core ie;

        // step 2. saving cache file
        std::map<std::string, std::string> deviceConfig;
        ie.SetConfig({ {CONFIG_KEY(CACHE_DIR), "./cache"}});

        // step 3. load IR / ONNX model
        CNNNetwork network = ie.ReadNetwork(model_path, bin_path);

        // step 4. get input & output format (allow to set input precision & get input / output layer name)
        InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
        InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();

        // resize 
        auto input_shapes = network.getInputShapes();
        std::string input_name;
        InferenceEngine::SizeVector input_shape;
        std::tie(input_name, input_shape) = *input_shapes.begin();
        input_shape[0] = 1;
        input_shape[1] = 3;
        input_shape[2] = input_width;
        input_shape[3] = input_height;
        input_shapes[input_name] = input_shape;
        network.reshape(input_shapes);

        //std::string input_name = inputs.begin()->first;
        inputs.begin()->second->setLayout(Layout::NCHW);
        inputs.begin()->second->setPrecision(Precision::FP32);
        inputs.begin()->second->getPreProcess().setColorFormat(ColorFormat::RGB);

        std::string output_name = outputs.begin()->first;
        outputs.begin()->second->setPrecision(Precision::FP32);

        // step 5. loading a model to the device
        ExecutableNetwork executable_network = ie.LoadNetwork(network, infer_device);

        // step 6. create an infer request
        InferRequest infer_request = executable_network.CreateInferRequest();
        infer_request = executable_network.CreateInferRequest();

        Blob::Ptr input_blob = infer_request.GetBlob(input_name);
        Blob::Ptr output_blob = infer_request.GetBlob(output_name);

        // step 7. prepare dummy input
        float* raw_data = new float[1 * 3 * input_height * input_width];
        //InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32, { 1, 3, 256, 256 }, InferenceEngine::Layout::NCHW);
        //Blob::Ptr imgBlob = InferenceEngine::make_shared_blob<float>(tDesc, raw_data);
        //infer_request.SetBlob(input_name, imgBlob);  // infer_request accepts input blob of any size

        // step 7. do inference
        // running the request synchronously
        infer_request.Infer();
        int i = 0;

        while (i < inference_times) {
            // input raw data to input blob
            float* input_tensor = static_cast<float*>(input_blob->buffer());
            uint32_t size = 1 * 3 * input_height * input_width;

            for (int i = 0; i < size; ++i) {
                input_tensor[i] = raw_data[i];
            }

            std::chrono::steady_clock::time_point start_time = std::chrono::high_resolution_clock::now();

            infer_request.Infer();
            i++;

            std::chrono::steady_clock::time_point end_time = std::chrono::high_resolution_clock::now();
            float time_count = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(end_time - start_time).count();
            if (time_count < sleep) {
                Sleep(sleep - time_count);
            }
            printf("\rTime: %d (ms)\n", (int)(time_count));
        }
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Finish! will close in 10 sec ..." << std::endl;
    Sleep(10000);
    return 0;
}

void UserInput() {
    // set model
    std::vector<std::filesystem::path> model_list;
    for (const auto& file : std::filesystem::recursive_directory_iterator("../")) {
        if (file.path().extension().string() == ".xml" && file.path().filename().string() != "plugins.xml") model_list.push_back(file);
    }

    if (!model_list.size()) {
        printf("\nCan not find any IR model (.xml) please put it in root folder\n");
        return;
    }
    else if (model_list.size() == 1) {        
        printf("\nFind model: %s", model_list[0].string().c_str());
        model_path = model_list[0].string();
    }
    else {
        std::cout << "\nFind model: ";
        for (int i = 0; i < model_list.size(); ++i) {
            printf("\n(%d): %s", i, model_list[i].string().c_str());
        }
        std::cout << "Choose model: ";
        std::getline(std::cin, user_input_str);
        model_id = (user_input_str == "") ? 0 : stoi(user_input_str);
        model_path = model_list[min(model_id, model_list.size())].string();
    }

    bin_path = model_path.substr(0, model_path.find_last_of('.')) + ".bin";

    // set model input shape
    std::cout << "\nInput model width  [default 256]: ";
    std::getline(std::cin, user_input_str);
    input_width = (user_input_str == "") ? 256 : stoi(user_input_str);

    std::cout << "Input model height [default 256]: ";
    std::getline(std::cin, user_input_str);
    input_height = (user_input_str == "") ? 256 : stoi(user_input_str);

    // Select adapter
    printf("\nUse CPU, GPU or VPUX to inference [default CPU]: ");
    std::getline(std::cin, user_input_str);
    infer_device = (user_input_str == "") ? "CPU" : user_input_str;

    printf("\nInference times (default 100): ");
    std::getline(std::cin, user_input_str);
    inference_times = (user_input_str == "") ? 100 : stoi(user_input_str);

    printf("\nSleep (ms) (default 0): ");
    std::getline(std::cin, user_input_str);
    sleep = (user_input_str == "") ? 0 : stoi(user_input_str);
    printf("\n");
}

void PrintConfig() {
    std::cout << "\nConfiguration: " << std::endl;
    std::cout << "Inference Adapter: " << infer_device << std::endl;
    std::cout << "Input Model (.xml): " << model_path << std::endl;
    std::cout << "Input Model (.bin): " << bin_path << std::endl;
    std::cout << "Model Input Shape: ( 1 x 3 x " << input_height << " x "<< input_width << ")" << std::endl;
    std::cout << "Inference times: " << inference_times << std::endl;
    std::cout << std::endl;
}

bool CheckExist(const std::string src) {
    struct stat buffer;
    return (stat(src.c_str(), &buffer) == 0);
}



