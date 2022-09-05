#include <iostream>
#include <string>
#include <numeric>
#include<sstream> 
#include <codecvt>
#include <cstdlib>

#include <Windows.h>
#include <wrl.h>

// D3D 
#include <d3d11.h>
#include "direct3d11.interop.h"

// WinRT
#include <winrt/Windows.Foundation.Collections.h>

// WinML includes
#include <winrt/Microsoft.AI.MachineLearning.h>
#include "Microsoft.AI.Machinelearning.Native.h" 

#include <filesystem>

using convert_type = std::codecvt_utf8<wchar_t>;
using wstring_to_utf8 = std::wstring_convert<convert_type, wchar_t>;



