#include "pch.h"

using namespace std;
using namespace winrt::Windows::Media;
using namespace winrt::Microsoft::AI::MachineLearning;

// get model path
winrt::hstring model_path;

int input_n = 1;
int input_c;
int input_h;
int input_w;
int infer_device;
int num_of_infer; 
bool reshape_model = false;
float break_time;
string user_input;

// windows machine learning
LearningModel model = nullptr;
LearningModelSession session = nullptr;
LearningModelDevice device = nullptr;

// timer
chrono::steady_clock::time_point start_time;
chrono::steady_clock::time_point end_time;
int latency;
vector<int> infer_time_array;

void UserInput();
void SelectAdapter();

// Entry
int main(int argc, char* argv[]) {
    UserInput();

    // Loading model
    start_time = chrono::high_resolution_clock::now();
    model = LearningModel::LoadFromFilePath(model_path);
    printf("Loading modelfile '%ws'\n", model_path.c_str());
    printf("Load WINML Model [SUCCEEDED]\n");

    // show and setting model input shape
    vector<int64_t> model_input_shape;
    auto feature_decript = model.InputFeatures().GetAt(0).as<TensorFeatureDescriptor>();
    for (int i = 0; i < feature_decript.Shape().Size(); ++i)
        model_input_shape.push_back(feature_decript.Shape().GetAt(i));
    
    printf("\nmodel shape: (");
    for (int i = 0; i < model_input_shape.size(); ++i) {
        (model_input_shape[i] == -1) ? printf("None") : printf("%d", model_input_shape[i]);
        if (i != model_input_shape.size() - 1)  printf(", ");
    }
    printf(")\n");

    for (int i = 0; i < model_input_shape.size(); ++i) {
        if (model_input_shape[i] == -1) {
            reshape_model = true;
            printf("Find undefined dimension in index %d, please enter dimension: ", i);
            int input_size;
            cin >> input_size;
            model_input_shape[i] = input_size;
        }
    }

    if (model_input_shape.size() == 4) {
        input_n = model_input_shape[0];
        if (model_input_shape[1] == 1 || model_input_shape[1] == 3) { // channel first
            input_c = model_input_shape[1];
            input_h = model_input_shape[2];
            input_w = model_input_shape[3];
        }
        else {  // channel last
            input_h = model_input_shape[1];
            input_w = model_input_shape[2];
            input_c = model_input_shape[3];
        }
    }
    else {
        if (model_input_shape[0] == 1 || model_input_shape[0] == 3) { // channel first
            input_c = model_input_shape[0];
            input_h = model_input_shape[1];
            input_w = model_input_shape[2];
        }
        else {  // channel last
            input_h = model_input_shape[0];
            input_w = model_input_shape[1];
            input_c = model_input_shape[2];
        }
    }
    
    if (reshape_model) {
        printf("\nNew model shape: (");
        for (int i = 0; i < model_input_shape.size(); ++i) {
            printf("%d", model_input_shape[i]);
            if (i != model_input_shape.size() - 1)  printf(", ");
        }
        printf(")\n");
    }

    // Create a session and binding
    LearningModelSessionOptions sessionOptions;
    // Define input dimensions to concrete values in order to achieve better runtime performance
    sessionOptions.OverrideNamedDimension(L"input_cx", input_h);
    sessionOptions.OverrideNamedDimension(L"input_cy", input_w);
    sessionOptions.BatchSizeOverride(input_n);
    session = LearningModelSession(model, device, sessionOptions);
    //session = LearningModelSession(model, device);
    LearningModelBinding binding(session);

    end_time = chrono::high_resolution_clock::now();
    latency = chrono::duration_cast<chrono::duration<double, ratio<1, 1000>>>(end_time - start_time).count();
    printf("Take %d (ms) for initialization\n", (int)(latency));

    uint32_t size = 1;
    for (auto i : model_input_shape) size *= i;
    float* raw_data = new float[size];
    int i = 0;

    while (i < num_of_infer) {
        start_time = chrono::high_resolution_clock::now();

        // Create WinML tensor float
        TensorFloat tf = TensorFloat::Create(model_input_shape);

        // Create memory for input array
        float* pCPUInputTensor = nullptr;
        uint32_t uInputCapacity;

        winrt::com_ptr<ITensorNative>  itn = tf.as<ITensorNative>();

        // Gets the tensor¡¦s buffer as an bytes array
        itn->GetBuffer(reinterpret_cast<BYTE**>(&pCPUInputTensor), &uInputCapacity);

        for (int i = 0; i < size; ++i) {
            pCPUInputTensor[i] = raw_data[i];
        }

        binding.Clear();
        auto&& description = model.InputFeatures().GetAt(0);

        // Create binding and then bind input features
        binding.Bind(description.Name(), tf);

        auto results = session.Evaluate(binding, L"");
        i++;

        end_time = chrono::high_resolution_clock::now();
        latency = chrono::duration_cast<chrono::duration<double, ratio<1, 1000>>>(end_time - start_time).count();

        if (i > 10) infer_time_array.push_back(int(latency));

        if (latency < break_time) {
            Sleep(break_time - latency);
        }
        printf("\rlatency: %d (ms)", (int)(latency));
    }
    printf("\n");
    printf("\nminium inference latency:  %d (ms)", *min_element(infer_time_array.begin(), infer_time_array.end()));
    printf("\nmaximum inference latenct: %d (ms)", *max_element(infer_time_array.begin(), infer_time_array.end()));
    printf("\naverage inference latency: %f (ms)", reduce(infer_time_array.begin(), infer_time_array.end(), 0.0) / infer_time_array.size());
}

void UserInput() {
    // Select adapter
    printf("Use CPU (0) or GPU (1) to inference [default CPU]: ");
    getline(cin, user_input);
    infer_device = (user_input == "") ? 0 : stoi(user_input);
    printf("\n");

    SelectAdapter();

    // set model
    vector<filesystem::path> model_list;
    for (const auto& file : filesystem::recursive_directory_iterator("./")) {
        if (file.path().extension().string() == ".onnx") model_list.push_back(file);
    }

    if (!model_list.size()) {
        printf("Can not find any model (.onnx) please put it in root folder\n");
        exit;
    }
    else if (model_list.size() == 1) {
        model_path = model_list[0].wstring();
    }
    else {
        cout << "Find model: \n";
        for (int i = 0; i < model_list.size(); ++i) {
            printf("(%d): %s\n", i, model_list[i].string().c_str());
        }
        cout << "Choose model [default 0]: ";
        getline(cin, user_input);
        model_path = (user_input == "") ? model_list[0].wstring() : model_list[min(stoi(user_input), model_list.size())].wstring();
        printf("\n");
    }

    printf("\nNumber iterations [default 100]: ");
    getline(cin, user_input);
    num_of_infer = (user_input == "") ? 100 : stoi(user_input);

    printf("\nSleep (ms) [default 0]: ");
    getline(cin, user_input);
    break_time = (user_input == "") ? 0 : stoi(user_input);
}

void SelectAdapter() {
    if (!infer_device)
        device = LearningModelDevice(LearningModelDeviceKind::Cpu);
    else {
        D3D_FEATURE_LEVEL FeatureLevels[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0, D3D_FEATURE_LEVEL_9_1 };
        UINT NumFeatureLevels = ARRAYSIZE(FeatureLevels);
        D3D_FEATURE_LEVEL FeatureLevel;

        UINT i = 0;
        IDXGIAdapter* pAdapter;
        vector <IDXGIAdapter*> vAdapters;
        IDXGIFactory1* pFactory = NULL;
        CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory);

        // Show all adapter
        while (pFactory->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND)
        {
            DXGI_ADAPTER_DESC adapter_desc;
            pAdapter->GetDesc(&adapter_desc);
            printf("Adapter %d: %ls\n", i, adapter_desc.Description);
            vAdapters.push_back(pAdapter);
            ++i;
        }
        printf("Choose adapter (0 - %d) [default 0]: ", i - 1);
        int select_adapter;
        getline(cin, user_input);
        select_adapter = (user_input == "") ? 0 : min(stoi(user_input), i - 1);
        printf("\n");

        // Create D3D11 Device
        winrt::com_ptr<ID3D11Device> m_infer_device;
        winrt::com_ptr<ID3D11DeviceContext> m_inference_context;
        D3D11CreateDevice(vAdapters[select_adapter], D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, FeatureLevels, NumFeatureLevels, D3D11_SDK_VERSION, m_infer_device.put(), &FeatureLevel, m_inference_context.put());

        IDXGIDevice* m_DxgiDevice = nullptr;
        m_infer_device->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&m_DxgiDevice));

        auto m_d3d_device = CreateDirect3DDevice(m_DxgiDevice);
        device = LearningModelDevice::CreateFromDirect3D11Device(m_d3d_device);
    }
}

