//
//  SpeechRecognizerTest.m
//  libtorch-ios
//
//  Created by Nagy Daniel on 2020. 02. 11..
//  Copyright © 2020. Nagy Daniel. All rights reserved.
//

#import "SpeechRecognizerTest.h"
#import <LibTorch/LibTorch.h>

@implementation SpeechRecognizer
{
 @protected
  torch::jit::script::Module _impl;
}

- (nullable instancetype)initWithModelFileAtPath:(NSString*)filePath {
    self = [super init];
    
    if (self) {
        try {
            auto qengines = at::globalContext().supportedQEngines();
            if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) != qengines.end()) {
                at::globalContext().setQEngine(at::QEngine::QNNPACK);
            }
            _impl = torch::jit::load(filePath.UTF8String);
            _impl.eval();
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}

+ (nullable void*)testPreprocessing:(void *)audioData {
    // TODO: Compute sepctrogram sequences from audio data;
    auto t = at::randn({1000});
    auto st = torch::stft(t, 96);
    return &st;
}
@end




