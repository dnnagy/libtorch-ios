//
//  ViewController.swift
//  libtorch-ios
//
//  Created by Nagy Daniel on 2020. 02. 11..
//  Copyright © 2020. Nagy Daniel. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        
        
//        if let filePath = Bundle.main.path(forResource: "cnnmodel", ofType: "pt") {
//            let C = 1
//            let H = 42
//            let W = 10
//            let N = 1
//            let T = 12
//            
//            // This model takes a tensor of shape (T, N, C, H, W)
//            var tensor = (1...T*N*C*H*W).map({_ in Float(Int.random(in: -255...255))/255.0})
//            let output = TorchTests.runModel(atFilePath: filePath, withTensorData: UnsafeMutableRawPointer(&tensor), ofShape: [T, N, C, H, W] as [NSNumber])
//            print(output)
//        }
//        TorchTests.testFFTPack()
        
        // Test CNN_LSTM_trained
        if let modelFilePath = Bundle.main.path(forResource: "CNN_LSTM_trained", ofType: "pt") {
            if let tensorFilePath = Bundle.main.path(forResource: "tensor-28", ofType: "bin") {
                // Load tensor data
                let file: FileHandle? = FileHandle(forReadingAtPath: tensorFilePath)
                
                // Read all the data
                let data = file!.readDataToEndOfFile()
                let pointer = UnsafeMutablePointer<UInt8>.allocate(capacity: data.count)
                data.copyBytes(to: pointer, count: data.count)
                let rawPointer = UnsafeMutableRawPointer(pointer)
                
                let output = TorchTests.runModel(atFilePath: modelFilePath, withTensorData: rawPointer, ofShape: [1, 16, 1, 49, 15])
                
                print("Shape:", output!["shape"])
                print("Data:", output!["data"])
            }
        }
        
    }
    
}

