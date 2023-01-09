#include "src/cpp/examples/gptpu_model_utils.h"

#include <memory>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"

//#include "tensorflow/lite/core/api/profiler.h"

#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <cassert>

namespace coral {
    struct Resource* resource;
    bool resource_valid = false;

    DeviceHandler::DeviceHandler(){
        if(resource_valid == false){
            resource_valid = true;
            resource = new struct Resource;
        }
    }

    DeviceHandler::~DeviceHandler(){
        // Reset all vectors to size zero.
        if(resource_valid == true){
            resource_valid = false;
            pthread_mutex_lock(&resource->mtx);
            if(resource->device_cleanup == false){
                resource->device_cleanup = true;
                resource->device_opened      =
                    std::vector<bool>();
                resource->interpreters       = 
                    std::vector<std::shared_ptr<tflite::Interpreter>>();
                resource->edgetpu_contexts   = 
                    std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>>();
                resource->enumerate_edgetpus = 
                    std::vector<edgetpu::EdgeTpuManager::DeviceEnumerationRecord>();
                resource->models             = 
                    std::vector<std::shared_ptr<tflite::FlatBufferModel>>();
            }
            pthread_mutex_unlock(&resource->mtx);
            pthread_mutex_destroy(&resource->mtx);
            delete resource;
        }
    }

    unsigned int DeviceHandler::list_devices(bool verbose){
        pthread_mutex_lock(&resource->mtx);
        if(resource->device_listed == false){
            resource->device_listed = true;
            resource->enumerate_edgetpus = 
                edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
            resource->total_dev_cnt = resource->enumerate_edgetpus.size();
            unsigned int total_dev_cnt = resource->total_dev_cnt;
            resource->edgetpu_contexts.resize(total_dev_cnt);
            resource->device_opened.resize(total_dev_cnt); // deafult to all false
            resource->models.resize(0); 
            resource->interpreters.resize(0);
            assert( resource->enumerate_edgetpus.size() == 
                resource->edgetpu_contexts.size() );
            if(verbose){
                std::cout << "enumerated edgetpu: " << std::endl;
                for(auto it = resource->enumerate_edgetpus.begin() ; it != resource->enumerate_edgetpus.end() ; it++){
                    std::cout << it->path << std::endl;
                }
            }
        }else{
            if(verbose){
                std::cout << __func__ 
                          << ": device listing skipped (was listed)" 
                          << std::endl;
            }
        }
        pthread_mutex_unlock(&resource->mtx);
        return resource->total_dev_cnt;
    }

    void DeviceHandler::open_device(unsigned int tpuid, bool verbose){
        assert(tpuid < resource->device_opened.size());
        if(resource->device_opened[tpuid] == false){
            resource->device_opened[tpuid] = true;
            resource->edgetpu_contexts[tpuid] = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
                resource->enumerate_edgetpus[tpuid].type,
                resource->enumerate_edgetpus[tpuid].path
        ); // skip opening the opened device with id 'tpuid'
        }
        if(verbose){
            std::cout << "opened device: ";
            std::cout << "type: " << resource->enumerate_edgetpus[tpuid].type;
            std::cout << ", path: " << resource->enumerate_edgetpus[tpuid].path;
            std::cout << std::endl;
        }
    }

    unsigned int DeviceHandler::build_model(const std::string& model_path){
        pthread_mutex_lock(&resource->mtx);
        std::shared_ptr<tflite::FlatBufferModel> local_model_tmp;
        local_model_tmp = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        if(local_model_tmp == nullptr){
            std::cout << "Fail to build FlatBufferModel from file: " << model_path << std::endl;
            std::abort();
        }
        unsigned int model_cnt = this->get_model_count();
        unsigned int model_id = model_cnt;
        model_cnt += 1;
        this->set_model_count(model_cnt);
        resource->models.resize(model_cnt);
        resource->interpreters.resize(model_cnt);
        resource->models[model_id] = local_model_tmp;
        pthread_mutex_unlock(&resource->mtx);
        return model_id;
    }

    /* Multiple diff model_id on the same tpuid is allowed.  */
    void DeviceHandler::build_interpreter(unsigned int tpuid, unsigned int model_id){
        pthread_mutex_lock(&resource->mtx);
        assert(tpuid < resource->device_opened.size());
        assert(tpuid < resource->edgetpu_contexts.size());
        std::shared_ptr<tflite::Interpreter> tmp;
        tmp = BuildEdgeTpuInterpreter(
            *resource->models[model_id],
            resource->edgetpu_contexts[tpuid].get()
        );
        if(tmp == nullptr){
            std::cout << "Fail to build interpreter ( ";
            std::cout << "tpu_id: " << tpuid;
            std::cout << ", model_id: " << model_id;
            std::cout << " )" << std::endl;
            std::abort();
        }
        assert(model_id < resource->interpreters.size());
        resource->interpreters[model_id] = tmp;
        pthread_mutex_unlock(&resource->mtx);
    }
                            
    void DeviceHandler::populate_input(uint8_t* data, int size, unsigned int model_id){
        assert(model_id < resource->interpreters.size());
        assert(resource->interpreters[model_id] != nullptr);
        uint8_t* input = resource->interpreters[model_id].get()->typed_input_tensor<uint8_t>(0);
        std::memcpy(input, data, size);
    }

    void DeviceHandler::model_invoke(unsigned int model_id, int iter){
        assert(model_id < resource->interpreters.size());
        assert(resource->interpreters[model_id] != nullptr);
        std::shared_ptr<tflite::Interpreter> tmp = resource->interpreters[model_id];
        for(int i = 0 ; i < iter ; i++){
            tmp.get()->Invoke();
        }
    }

    void DeviceHandler::get_output(int* data, unsigned int model_id){
        assert(model_id < resource->interpreters.size());
        assert(resource->interpreters[model_id] != nullptr);
        std::shared_ptr<tflite::Interpreter> tmp = resource->interpreters[model_id];
        const auto& output_indices = tmp->outputs();
        const int num_outputs = output_indices.size();
        int num_values = 0;
        for(int i = 0 ; i < num_outputs ; ++i){
            const auto* out_tensor = tmp->tensor(output_indices[i]);
            assert(out_tensor != nullptr);
            if(out_tensor->type == kTfLiteUInt8){
                num_values = out_tensor->bytes;
                const uint8_t* output = tmp->typed_output_tensor<uint8_t>(i);
                for(int j = 0 ; j < num_values ; ++j){
                    data[j] = (output[j] - out_tensor->params.zero_point) *
                        out_tensor->params.scale;
                }
            }else{
                std::cout << "Tensor " << out_tensor->name
                          << "has unsupported output type: " << out_tensor->type
                          << std::endl;
                std::abort();
            }
        }
    }
    
    /*   
            Caller has to pre-allocate it's output data array in uint8_t pointer 
            type before calling this function.
        
            Caller has it's own full control over the output array since it's 
            conceptually copied from the internal interpreter. 
         
            This new output design avoids unnessacery data copy and the latency is 
            reduced 100x to be minor compared to the main model invokation time.
    */
    void DeviceHandler::get_raw_output(uint8_t* data, int size, unsigned int model_id, uint8_t& zero_point, float& scale){
        assert(model_id < resource->interpreters.size());
        assert(resource->interpreters[model_id] != nullptr);
        std::shared_ptr<tflite::Interpreter> tmp = resource->interpreters[model_id];
        const auto& output_indices = tmp->outputs();
        const int num_outputs = output_indices.size();
        int num_values = 0;
        for(int i = 0 ; i < num_outputs ; ++i){
            const auto* out_tensor = tmp->tensor(output_indices[i]);
            assert(out_tensor != nullptr);
            if(out_tensor->type == kTfLiteUInt8){
                num_values = out_tensor->bytes;
                std::memcpy(data,
                            tmp->typed_output_tensor<uint8_t>(i),
                            size);
                zero_point = out_tensor->params.zero_point;
                scale      = out_tensor->params.scale;
            }else{
                std::cout << "Tensor " << out_tensor->name
                          << "has unsupported output type: " << out_tensor->type
                          << std::endl;
                std::abort();
            }
        }
    }

    void DeviceHandler::set_model_count(unsigned int val){ resource->total_model_cnt = val; }
    unsigned int DeviceHandler::get_model_count(){return resource->total_model_cnt; }

std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* edgetpu_context) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
    std::cerr << "Failed to build interpreter." << std::endl;
  }
  // Bind given context with interpreter.
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
  interpreter->SetNumThreads(2);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }
  return interpreter;
}

std::vector<float> RunInference(const std::vector<uint8_t>& input_data,
                                tflite::Interpreter* interpreter) {
  std::vector<float> output_data;
  uint8_t* input = interpreter->typed_input_tensor<uint8_t>(0);
  std::memcpy(input, input_data.data(), input_data.size());

  interpreter->Invoke();

  //std::cerr << " ===== My insertion code =====" << std::endl;
  //auto profiler = interpreter->GetProfiler();
  //std::cerr << "profiler type: " << typeid(profiler).name() << std::endl;
  //std::cerr << profiler << std::endl;  
  //interpreter->SetProfiler(profiler);
  //std::cerr << " ===== My insertion code =====" << std::endl;

  const auto& output_indices = interpreter->outputs();
  const int num_outputs = output_indices.size();
  int out_idx = 0;
  for (int i = 0; i < num_outputs; ++i) {
    const auto* out_tensor = interpreter->tensor(output_indices[i]);
    assert(out_tensor != nullptr);
    if (out_tensor->type == kTfLiteUInt8) {
      const int num_values = out_tensor->bytes;
      output_data.resize(out_idx + num_values);
      const uint8_t* output = interpreter->typed_output_tensor<uint8_t>(i);
      for (int j = 0; j < num_values; ++j) {
        output_data[out_idx++] = (output[j] - out_tensor->params.zero_point) *
                                 out_tensor->params.scale;
      }
    } else if (out_tensor->type == kTfLiteFloat32) {
      const int num_values = out_tensor->bytes / sizeof(float);
      output_data.resize(out_idx + num_values);
      const float* output = interpreter->typed_output_tensor<float>(i);
      for (int j = 0; j < num_values; ++j) {
        output_data[out_idx++] = output[j];
      }
    } else {
      std::cerr << "Tensor " << out_tensor->name
                << " has unsupported output type: " << out_tensor->type
                << std::endl;
    }
  }
  return output_data;
}

std::array<int, 3> GetInputShape(const tflite::Interpreter& interpreter,
                                 int index) {
  const int tensor_index = interpreter.inputs()[index];
  const TfLiteIntArray* dims = interpreter.tensor(tensor_index)->dims;
  return std::array<int, 3>{dims->data[1], dims->data[2], dims->data[3]};
}

}  // namespace coral
