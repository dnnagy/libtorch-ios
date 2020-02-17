//
//  SpeechRecognizerTest.h
//  libtorch-ios
//
//  Created by Nagy Daniel on 2020. 02. 11..
//  Copyright © 2020. Nagy Daniel. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface SpeechRecognizer : NSObject

- (nullable instancetype)initWithModelFileAtPath:(NSString*)filePath
    NS_SWIFT_NAME(init(modelFile:)) NS_DESIGNATED_INITIALIZER;
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

+ (nullable void*)testPreprocessing:(void*)audioData
__attribute__((swift_name("testPreprocessing(audioData:)")));
@end

NS_ASSUME_NONNULL_END
