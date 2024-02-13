//===- HardwareConfig.cpp - Hardware configuration ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ManagedStatic.h"
#include "mlir/Support/SystemDesc.h"

using namespace llvm;
using namespace mlir;

//ManagedStatic<SystemDesc> systemDesc; 

LogicalResult SystemDesc::readSystemDescFromJSONFile(llvm::StringRef filename) {
  #if 0
  std::string errorMessage;
  auto file = openInputFile(filename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }
  #endif

  // Code to parse here

  // set entries
  auto desc = mlir::DeviceDesc(0, mlir::DeviceDesc::CPU)
                    .setDescription("Intel Xeon 8480")
                    .setProperty("L1_CACHE_SIZE_IN_BYTES", 8192);
  SystemDesc::getGlobalSystemDesc()->addDeviceDesc(desc);

  return success();
}

int SystemDesc::getCPUL1CacheSizeInBytes(DeviceDesc::DeviceID deviceID) {
  return SystemDesc::getGlobalSystemDesc()->getDeviceDesc(deviceID)
                    .getPropertyValueAsInt("L1_CACHE_SIZE_IN_BYTES");
}
