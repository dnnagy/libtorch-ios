//
//  TorchModule.h
//  libtorch-ios
//
//  Created by Nagy Daniel on 2020. 02. 11..
//  Copyright © 2020. Nagy Daniel. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface TorchTests : NSObject

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

// Test torch C++
+ (nullable NSArray<NSNumber*>*)testAddRandomTensors
__attribute__((swift_name("testAddRandomTensors()")));

+ (nullable NSArray<NSNumber*>*)runModelAtFilePath:(NSString*)filePath withTensorData:(void*) data ofShape:(NSArray<NSNumber*>*) shape
__attribute__((swift_name("runModelAt(filePath:, data:, shape:)")));

+ (nullable void*)testSTFT
__attribute__((swift_name("testSTFT()")));

+ (nullable void*)testAutograd
__attribute__((swift_name("testAutograd()")));

+ (nullable void*)testNeuralNetworks
__attribute__((swift_name("testNeuralNetworks()")));

@end

NS_ASSUME_NONNULL_END