#include <algorithm>
#include <cmath>
#include <chrono>  // NOLINT
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <sys/mman.h>
#include <ostream>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <string>
#include <time.h>
#include <iomanip>
#include "src/cpp/examples/gptpu_utils.h"
#include "src/cpp/examples/gptpu_model_utils.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gptpu_utils{

/* To make sure that multiple DeviceHanlder instances are minapulating 
   the same underlying coral::DeviceHandler* device_handler
    
    Globally there should be only one device_handler. But multiple are allowed.
*/
coral::DeviceHandler* device_handler = nullptr;

EdgeTpuHandler::EdgeTpuHandler(){
    if(device_handler == nullptr){
       device_handler = new coral::DeviceHandler;
    }
}

EdgeTpuHandler::~EdgeTpuHandler(){
    if(device_handler != nullptr){
        delete device_handler;
    }
}

unsigned int EdgeTpuHandler::list_devices(bool verbose){
    return device_handler->list_devices(verbose);
}

void EdgeTpuHandler::open_device(unsigned int tpuid, bool verbose){
    device_handler->open_device(tpuid, verbose);
}

unsigned int EdgeTpuHandler::build_model(const std::string& model_path){
    return device_handler->build_model(model_path);
}

void EdgeTpuHandler::build_interpreter(unsigned int tpuid, unsigned int model_id){
    device_handler->build_interpreter(tpuid, model_id);
}

void EdgeTpuHandler::populate_input(uint8_t* data, int size, unsigned int model_id){
    device_handler->populate_input(data, size, model_id);
}

void EdgeTpuHandler::model_invoke(unsigned int model_id, int iter){
    device_handler->model_invoke(model_id, iter);
}

void EdgeTpuHandler::get_output(int* data, unsigned int model_id){
    device_handler->get_output(data, model_id);
}

void EdgeTpuHandler::get_raw_output(uint8_t* data, int size, unsigned int model_id, uint8_t& zero_point, float& scale){
    device_handler->get_raw_output(data, size, model_id, zero_point, scale);
}
}// end namespace
