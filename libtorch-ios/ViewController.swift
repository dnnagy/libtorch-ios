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
        
        
        if let filePath = Bundle.main.path(forResource: "cnnmodel", ofType: "pt") {
            let C = 1
            let H = 42
            let W = 10
            let N = 1
            let T = 12
            
            // This model takes a tensor of shape (T, N, C, H, W)
            var tensor = (1...T*N*C*H*W).map({_ in Float(Int.random(in: -255...255))/255.0})
            let output = TorchTests.runModel(atFilePath: filePath, withTensorData: UnsafeMutableRawPointer(&tensor), ofShape: [T, N, C, H, W] as [NSNumber])
            print(output)
        }
        
    }
    
}

