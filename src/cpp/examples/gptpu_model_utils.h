#ifndef EDGETPU_CPP_EXAMPLES_GPTPU_MODEL_UTILS_H_
#define EDGETPU_CPP_EXAMPLES_GPTPU_MODEL_UTILS_H_

#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <pthread.h>

#include "edgetpu.h"
//#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

namespace coral {
    struct Resource {
        // Global resources across multiple DeviceHander instances.
        pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
        bool device_listed = false;
        bool device_cleanup = false;
        unsigned int total_dev_cnt = 0; // initially as 0 (avilable device count not checked yet)
        unsigned int total_model_cnt = 0;
        std::vector<bool> device_opened;
        std::vector<std::shared_ptr<tflite::Interpreter>> interpreters;
        std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>> edgetpu_contexts;
        std::vector<edgetpu::EdgeTpuManager::DeviceEnumerationRecord> enumerate_edgetpus;
        std::vector<std::shared_ptr<tflite::FlatBufferModel>> models;
    };

    class DeviceHandler{
    public:
        DeviceHandler();
        ~DeviceHandler();

        // List all edgetpu devices and returns the number of them.
        unsigned int list_devices(bool verbose);

        // Open 'tpuid' edgetpu device.
        void open_device(unsigned int tpuid, bool verbose);

        // Separate out the original 'BuildInterpreter' into 'build_model' and 'build_interpreter' two functions
        // Build edgetpu model with an unique model id.
        // This is a thread-safe and atomic function.
        unsigned int build_model(const std::string& model_path);

        // Build edgetpu interpreter.
        void build_interpreter(unsigned int tpuid, unsigned int model_id);

        // Separate out the original 'RunInfernece' into populate_input invoke populate_output
        // Populate input array.
        void populate_input(uint8_t* data, int size, unsigned int model_id);

        // Actual invoke calling.
        void model_invoke(unsigned int model_id, int iter);

        // Get output array.
        void get_output(int* data, unsigned int model_id);
        
        // Get uin8_t raw output array.
        // It's caller's duty to make sure size is valid for data pointer.
        void get_raw_output(uint8_t* data, int size, unsigned int model_id, uint8_t& zero_point, float& scale);

    private:
        /* This two private helper APIs have to be protected by mutex lock. */
        void set_model_count(unsigned int size);
        unsigned int get_model_count();
    };

// Builds tflite Interpreter capable of running Edge TPU model.
std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model,
    edgetpu::EdgeTpuContext* edgetpu_context);

// Runs inference using given `interpreter`
std::vector<float> RunInference(const std::vector<uint8_t>& input_data, tflite::Interpreter* interpreter);

// Returns input tensor shape in the form {height, width, channels}.
std::array<int, 3> GetInputShape(const tflite::Interpreter& interpreter,
                                 int index);

}  // namespace coral
#endif  // EDGETPU_CPP_EXAMPLES_GPTPU_MODEL_UTILS_H_
