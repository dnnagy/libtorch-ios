//
//  ViewController.swift
//  libtorch-ios
//
//  Created by Nagy Daniel on 2020. 02. 11..
//  Copyright © 2020. Nagy Daniel. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    
    func testCNNBaseline() {
        let modelFilePath = Bundle.main.path(forResource: "CNN_LSTM_trained", ofType: "pt")!
        
        let tensorFilePath = Bundle.main.path(forResource: "tensor-28", ofType: "bin")!
        let file: FileHandle? = FileHandle(forReadingAtPath: tensorFilePath)
        let data = file!.readDataToEndOfFile()
        let pointer = UnsafeMutablePointer<UInt8>.allocate(capacity: data.count)
        data.copyBytes(to: pointer, count: data.count)
        let rawPointer = UnsafeMutableRawPointer(pointer)
             
        let output = TorchTests.runModel(atFilePath: modelFilePath, withTensorData: rawPointer, ofShape: [1, 16, 1, 49, 15],
                                         applyingArgmaxOnDim: 1)
             
        print("Shape:", output!["shape"])
        print("Data:", output!["data"])
    }
    
    func testWav2Cat() {
        let modelFilePath = Bundle.main.path(forResource: "Wav2Cat_baseline_scripted", ofType: "pt")!
        
        let tensorFilePath = Bundle.main.path(forResource: "label=7.0-len=5598", ofType: "bin")!
        let file: FileHandle? = FileHandle(forReadingAtPath: tensorFilePath)
        let data = file!.readDataToEndOfFile()
        let pointer = UnsafeMutablePointer<UInt8>.allocate(capacity: data.count)
        data.copyBytes(to: pointer, count: data.count)
        let rawPointer = UnsafeMutableRawPointer(pointer)
        
        let output = TorchTests.runModel(atFilePath: modelFilePath, withTensorData: rawPointer, ofShape: [1, 1, 5598], applyingArgmaxOnDim: 1)
        
        print("Shape:", output!["shape"])
        print("Data:", output!["data"])
    }
    override func viewDidLoad() {
        super.viewDidLoad()
        
        testWav2Cat()
        //testCNNBaseline()
    }
    
}

