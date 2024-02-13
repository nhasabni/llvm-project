//===- SYSTEMDESC.h - class to represent hardware configuration --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hardware configuration provides commonly used hardware information to different
// users, such as optimization passes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_SYSTEMDESC_H
#define MLIR_SUPPORT_SYSTEMDESC_H

#include <map>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/FileUtilities.h"

/// Sytem description file contains a list of device descriptions that
/// each describe a device (e.g., CPU, GPU, ASIC, etc.) in the system.
/// Example:
/// [
///  {
///    "ID": 1,
///    "TYPE": "CPU",
///    "DESCRIPTION": "Intel Xeon 8480",
///    "L1_CACHE_SIZE_IN_BYTES": 8192,
///    ...
///  },
///  {
///  
///  },
///  ...
/// ]
namespace mlir {

/// Describes the individual device from the system description
class DeviceDesc {
  public:
    /// Some typedefs
    using DeviceID = uint32_t;
    using DevicePropertyName = std::string;
    struct DevicePropertyValue {
      enum Tag {
        INT,
        FLOAT
      } tag;
      union {
        int iValue;
        float fValue;
      } data;

      bool operator == (const mlir::DeviceDesc::DevicePropertyValue& rhs) const {
        return tag == rhs.tag && data.iValue == rhs.data.iValue;
      }
      bool operator != (const mlir::DeviceDesc::DevicePropertyValue& rhs) const {
        return !(*this == rhs);
      }
    };
    using DevicePropertiesMapTy = std::map<DevicePropertyName, DevicePropertyValue>;

    typedef enum {
      CPU,
      GPU,
      SPECIAL
    } DeviceType;

    /// Basic constructor
    DeviceDesc() = delete;
    DeviceDesc(DeviceID id, DeviceType type) : ID(id), type(type) {}
    bool operator == (const mlir::DeviceDesc& rhs) const {
      return ID == rhs.getID() &&
             type == rhs.getType() &&
             deviceProperties == rhs.getProperties();
    }
    bool operator != (const mlir::DeviceDesc& rhs) const { return !(*this == rhs); }

    /// Set description
    DeviceDesc& setDescription(std::string desc) { description = desc; return *this;}
    /// Set property
    DeviceDesc& setProperty(llvm::StringRef name, int iv) {
      DevicePropertyValue value; value.tag = DevicePropertyValue::Tag::INT; value.data.iValue = iv;
      auto inserted = deviceProperties.insert(std::make_pair(std::string(name), value));
      if (!inserted.second && inserted.first->second != value) {
        llvm::report_fatal_error("Duplicate device property name found:" + name);
      }
      return *this;
    }
    DeviceDesc& setProperty(llvm::StringRef name, float fv) {
      DevicePropertyValue value; value.tag = DevicePropertyValue::Tag::FLOAT; value.data.fValue = fv;
      auto inserted = deviceProperties.insert(std::make_pair(std::string(name), value));
      if (!inserted.second && inserted.first->second != value) {
        llvm::report_fatal_error("Duplicate device property name found:" + name);
      }
      return *this;
    }
    /// Get ID
    DeviceID getID() const { return ID; }
    /// Get device type
    DeviceType getType() const { return type; }
    /// Get device description
    std::string getDescription() const { return description; }
    /// Get all of device properties
    const DevicePropertiesMapTy& getProperties() const { return deviceProperties; }
    /// Get property value: returns the value of the property with given name, if it exists.
    /// Otherwise throws exception (TODO)
    int getPropertyValueAsInt(llvm::StringRef name) const {
      // check that property with the given name exists
      auto iter = deviceProperties.find(std::string(name));
      if (iter == deviceProperties.end()) {
        llvm::report_fatal_error("Specified device property name not found:" + name);
      }
      // TODO: we can do a tag check here.
      return iter->second.data.iValue;
    }
    float getPropertyValueAsFloat(llvm::StringRef name) const {
      // check that property with the given name exists
      auto iter = deviceProperties.find(std::string(name));
      if (iter == deviceProperties.end()) {
        llvm::report_fatal_error("Specified device property name not found:" + name);
      }
      // TODO: we can do a tag check here.
      return iter->second.data.fValue;
    }

    /// Special functions
    auto getAllDevicePropertyNames() const {
      return llvm::map_range(
        deviceProperties,
        [](const DevicePropertiesMapTy::value_type &item) -> llvm::StringRef { return item.first; });
    }

  private:
    /// Unique device ID for every device
    DeviceID ID;

    /// Type of device
    DeviceType type;

    /// Some description of the device
    std::string description;
    
    /// Dictionary to store rest of the properties
    DevicePropertiesMapTy deviceProperties;
};

class SystemDesc {
  public:
    /// Singleton class - we want single system descriptor per program invocation
    static SystemDesc* getGlobalSystemDesc() {
      static SystemDesc gDesc;
      return &gDesc;
    }

    /// Read and parse system description from JSON file
    static LogicalResult readSystemDescFromJSONFile(llvm::StringRef filename);
    static void writeSystemDescToJSONFile(llvm::StringRef filename);

    /// Insert a new device description
    SystemDesc& addDeviceDesc(const DeviceDesc& desc) {
      auto inserted = deviceDescs.insert(std::make_pair(desc.getID(), desc));
      if (!inserted.second || inserted.first->second != desc) {
        llvm::report_fatal_error("Duplicate device description for ID:" +
          llvm::StringRef(std::to_string(desc.getID())));
      }
      return *getGlobalSystemDesc();
    }
    /// Get a device description
    const DeviceDesc& getDeviceDesc(DeviceDesc::DeviceID deviceID) {
      auto iter = deviceDescs.find(deviceID);
      if (iter != deviceDescs.end()) {
        return iter->second;
      }
      llvm::report_fatal_error("Device description with ID not found:" +
          llvm::StringRef(std::to_string(deviceID)));
    }

    /// Types
    using DeviceDescsMapTy = std::map<DeviceDesc::DeviceID, DeviceDesc>;

    // Generic functions: TODO
    /// Get number of CPU devices in the system
    static uint32_t getNumCPUDevices() { return 0; }
    static uint32_t getNumGPUDevices() { return 0; }

    // Device specific interface
    static int getCPUL1CacheSizeInBytes(DeviceDesc::DeviceID deviceID);

  private:
    SystemDesc() {}
    SystemDesc(const SystemDesc &) = delete;
    void operator=(const SystemDesc&) = delete;

  private:
    /// Map to store all the device descriptions
    DeviceDescsMapTy deviceDescs;
};

} // namespace mlir
#endif // MLIR_SUPPORT_SYSTEMDESC_H

