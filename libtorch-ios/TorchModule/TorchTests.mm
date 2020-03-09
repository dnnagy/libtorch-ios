//
//  TorchModule.m
//  libtorch-ios
//
//  Created by Nagy Daniel on 2020. 02. 11..
//  Copyright © 2020. Nagy Daniel. All rights reserved.
//

#import "TorchTests.h"
#import <LibTorch/LibTorch.h>
#import <algorithm>
#import <cstdlib>
#import "fftpack.h"


@implementation TorchTests

+ (nullable NSArray<NSNumber*>*)testAddRandomTensors{
    torch::Tensor t1 = torch::randint(/*low=*/1, /*high=*/10, {5, 5});
    torch::Tensor t2 = torch::randint(/*low=*/1, /*high=*/10, {5, 5});
    
    auto sum = t1+t2;
    
    std::cout << sum.sizes() << std::endl;
    std::cout << sum.numel() << std::endl;
    
    // Convert torcg Tensor to NSArray
    float* floatBuffer = sum.data_ptr<float>();
    if (!floatBuffer) {
      return nil;
    }
    NSMutableArray* results = [[NSMutableArray alloc] init];
    for (int i = 0; i < 25 ; i++) {
      [results addObject:@(floatBuffer[i])];
    }
    return [results copy];
}

+ (nullable void*)testSTFT {
    auto x = torch::randint(-255, 255, {22500})/255.0;
    auto S = torch::stft(x, 256);
    std::cout << at::stft(x, 256).sizes() << std::endl;
    std::cout << S.sizes() << std::endl;
    return nil;
}

+ (nullable void*)testAutograd {
    return nil;
}

+ (nullable void*)testNeuralNetworks {
    return nil;
}

void print_array(float* arr, int n){
    std::cout << "[";
    for(int k=0; k<n-1; k++){
        std::cout<<arr[k] << ", ";
    }
    std::cout << arr[n-1] << "]" << std::endl;
}

+ (nullable void*)testFFTPack {
    
    int n = 9;
    float wsave[2*n + 15];
    int ifac[n+15];
    
    float arr[9] = {11.0, 3.0, 4.05, 9.0, 10.3, 8.0, 4.934, 5.11, 9.12};
    
    
    // initialize  rfftf and rfftb
    __ogg_fdrffti(n, wsave, ifac);
    
    // forward transform of a real periodic sequence
    __ogg_fdrfftf(n, arr, wsave, ifac);
    
    print_array(arr, n);
    
    return nil;
}

+ (nullable NSDictionary*)runModelAtFilePath:(NSString*)filePath withTensorData:(void*) data ofShape:(NSArray*) shape {
    
    torch::jit::script::Module impl;
    
    try {
        auto qengines = at::globalContext().supportedQEngines();
        if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) != qengines.end()) {
            at::globalContext().setQEngine(at::QEngine::QNNPACK);
        }
        
        impl = torch::jit::load(filePath.UTF8String);
        impl.eval();
        
        // Construct torch tensor from data
        int64_t arr[shape.count];
        for(int i=0; i < shape.count; i++) {
           arr[i] = [[shape objectAtIndex:i] unsignedIntValue];
        }
        
        torch::Tensor tensor = torch::from_blob(data, at::IntArrayRef(arr, shape.count), torch::kFloat);
        
        // Run the model
        torch::autograd::AutoGradMode guard(false);
        at::AutoNonVariableTypeMode non_var_type_mode(true);
        auto outputTensor = impl.forward({tensor}).toTensor();
        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
          return nil;
        }
        
        NSMutableArray* results = [[NSMutableArray alloc] init];
        for (int i = 0; i < outputTensor.numel() ; i++) {
          [results addObject:@(floatBuffer[i])];
        }
        
        NSMutableArray* output_shape = [[NSMutableArray alloc] init];
        for (int i=0; i < outputTensor.sizes().size(); i++){
            [output_shape addObject:@(outputTensor.sizes().at(i))];
        }
        return @{ @"data": [results copy], @"shape": [output_shape copy] };
    } catch (const std::exception& e) {
        NSLog(@"%s", e.what());
    }
    return nil;
}

+ (nullable NSDictionary*)runModelAtFilePath:(NSString*)filePath withTensorData:(void*) data ofShape:(NSArray<NSNumber*>*) shape applyingArgmaxOnDim:(uint64_t) argmaxDim {
    
    torch::jit::script::Module impl;
    
    try {
        auto qengines = at::globalContext().supportedQEngines();
        if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) != qengines.end()) {
            at::globalContext().setQEngine(at::QEngine::QNNPACK);
        }
        
        impl = torch::jit::load(filePath.UTF8String);
        impl.eval();
        
        // Construct torch tensor from data
        int64_t arr[shape.count];
        for(int i=0; i < shape.count; i++) {
           arr[i] = [[shape objectAtIndex:i] unsignedIntValue];
        }
        
        
        // Disable autograd
        torch::autograd::AutoGradMode guard(false);
        at::AutoNonVariableTypeMode non_var_type_mode(true);
        
        // Input tensor
        torch::Tensor tensor = torch::from_blob(data, at::IntArrayRef(arr, shape.count), torch::kFloat);
        std::cout << "Input tensor shape = " << tensor.sizes() << std::endl;
        
        auto outputTensor = impl.forward({tensor}).toTensor();
        
        // Apply argmax
        outputTensor = at::argmax(outputTensor, argmaxDim);
        int64_t* buffer = outputTensor.data_ptr<int64_t>();
        if (!buffer) {
          return nil;
        }
        
        NSMutableArray* results = [[NSMutableArray alloc] init];
        for (int i = 0; i < outputTensor.numel() ; i++) {
          [results addObject:@(buffer[i])];
        }
        
        NSMutableArray* output_shape = [[NSMutableArray alloc] init];
        for (int i=0; i < outputTensor.sizes().size(); i++){
            [output_shape addObject:@(outputTensor.sizes().at(i))];
        }
        return @{ @"data": [results copy], @"shape": [output_shape copy] };
    } catch (const std::exception& e) {
        NSLog(@"%s", e.what());
    }

    return nil;
}
@end
