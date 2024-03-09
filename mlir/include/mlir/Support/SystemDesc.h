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
#include "llvm/Support/JSON.h"
#include "mlir/IR/MLIRContext.h"
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

    /// Type converters
    static DeviceID strToDeviceID(const std::string& id_str) {
      llvm::Expected<int64_t> id = llvm::json::parse<int64_t>(id_str);
      if (!id)
        llvm::report_fatal_error("Value of \"ID\" is not int");
      return static_cast<DeviceID>(id.get());
    }
    static DeviceType strToDeviceType(const std::string& type_str) {
      if (type_str == "CPU") return DeviceType::CPU;
      else if (type_str == "GPU") return DeviceType::GPU;
      else if (type_str == "SPECIAL") return DeviceType::SPECIAL;
      llvm::report_fatal_error("Value of \"Type\" is not CPU, GPU, or SPECIAL");
    }

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
    // We provide convenience interface to handle int/float value as string
    DeviceDesc& setProperty(llvm::StringRef name, const std::string& json_value) {
      // int64_t because llvm::json has int64_t support (not int)
      llvm::Expected<int64_t> iv = llvm::json::parse<int64_t>(json_value);
      if (iv) {
        *this = this->setProperty(name, static_cast<int>(iv.get()));
        return *this;
      }

      // Int type failed, try float now.
      // double because llvm::json has double support (not float)
      llvm::Expected<double> dv = llvm::json::parse<double>(json_value);
      if (dv) {
        *this = this->setProperty(name, static_cast<float>(dv.get()));
        return *this;
      }

      llvm::report_fatal_error("Neither int/float value in Device Description: key" + name);
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

    /// We use a list of key-value pairs to represent a system description in JSON.
    using DeviceDescJSONTy = std::map<std::string, std::string>;
    static DeviceDesc parseDeviceDescFromJSON(const DeviceDescJSONTy& device_desc);

    // -----------------------------------------------------------------------
    //          CPU specific methods
    // -----------------------------------------------------------------------
    static constexpr llvm::StringRef getCPUL1CacheSizeInBytesKeyName() {
      return "L1_CACHE_SIZE_IN_BYTES";
    }
    static constexpr llvm::StringRef getConvAndMatMulBlockingFactorKeyName() {
      return "CONV_AND_MATMUL_BLOCKING_FACTOR";
    }
    static constexpr llvm::StringRef getMatMulTileSizeInBytesKeyName() {
      return "MATMUL_TILE_SIZE_IN_BYTES";
    }

    size_t getL1CacheSizeInBytes() const {
      return (size_t) this->getPropertyValueAsInt(
                        DeviceDesc::getCPUL1CacheSizeInBytesKeyName());
    }
    void setL1CacheSizeInBytes(size_t value) {
      // Temporarily use int override until we support size_t
      this->setProperty(DeviceDesc::getCPUL1CacheSizeInBytesKeyName(), (int) value);
    }
    size_t getConvAndMatMulBlockingFactor() const {
      return (size_t) this->getPropertyValueAsInt(
                        DeviceDesc::getConvAndMatMulBlockingFactorKeyName());
    }
    void setConvAndMatMulBlockingFactor(size_t value) {
      // Temporarily use int override until we support size_t
      this->setProperty(DeviceDesc::getConvAndMatMulBlockingFactorKeyName(), (int) value);
    }
    size_t getMatMulTileSizeInBytes() const {
      return (size_t) this->getPropertyValueAsInt(
                        DeviceDesc::getMatMulTileSizeInBytesKeyName());
    }
    void setMatMulTileSizeInBytes(size_t value) {
      // Temporarily use int override until we support size_t
      this->setProperty(DeviceDesc::getMatMulTileSizeInBytesKeyName(), (int) value);
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
    SystemDesc() = default;

    /// Read and parse system description from JSON file
    LogicalResult readSystemDescFromJSONFile(llvm::StringRef filename);
    void writeSystemDescToJSONFile(llvm::StringRef filename);

    /// Insert a new device description
    SystemDesc& addDeviceDesc(const DeviceDesc& desc) {
      auto inserted = deviceDescs.insert(std::make_pair(desc.getID(), desc));
      if (!inserted.second || inserted.first->second != desc) {
        llvm::report_fatal_error("Duplicate device description for ID:" +
          llvm::StringRef(std::to_string(desc.getID())));
      }
      return *this;
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

  private:
    SystemDesc(const SystemDesc &) = delete;
    void operator=(const SystemDesc&) = delete;

  private:
    /// Map to store all the device descriptions
    DeviceDescsMapTy deviceDescs;
};

// An abstract class that represent device description for an abstract base device
//
// This class specifies minimum set of device properties that must be specified by
// the default device descriptor that will be used in case a user does not specify
// its own properties for the device.
class DefaultBaseDeviceDesc {
 public:
  virtual ~DefaultBaseDeviceDesc() {}
  virtual void registerDeviceDesc(MLIRContext *context) const = 0;

  /// -----------------------------------------------------------------------
  /// Set of common parameters of system description
  /// -----------------------------------------------------------------------
  // These methods allow to provide default values of these properties.
  virtual void setL1CacheSizeInBytes() = 0;

  /// Set of common questions asked by various passes
  // Blocking factor and tile size are typically used by tile/block passes.
  virtual void setConvAndMatMulBlockingFactor() = 0;
  virtual void setMatMulTileSize() = 0;
};

// Class that represent device description for a typical CPU device
class DefaultCPUDeviceDesc : public DefaultBaseDeviceDesc {
 public:
  // We use default ID of 0 because we are expecting to have only one device so far.
  // Not heterogeneous setup.
  DefaultCPUDeviceDesc() : cpu_device_desc(DeviceDesc(/* id */ 0, DeviceDesc::CPU)) {
    // Register all system properties
    this->setL1CacheSizeInBytes();
    this->setConvAndMatMulBlockingFactor();
    this->setMatMulTileSize();
  }

  ~DefaultCPUDeviceDesc() {}
  
  void registerDeviceDesc(MLIRContext *context) const override {
    context->getSystemDesc().addDeviceDesc(cpu_device_desc);
  }

  // -------------------------------------------------------------------------

  void setL1CacheSizeInBytes() override {
    cpu_device_desc.setL1CacheSizeInBytes(8192);
  }
  void setConvAndMatMulBlockingFactor() override {
    cpu_device_desc.setConvAndMatMulBlockingFactor(32);
  }
  void setMatMulTileSize() override {
    cpu_device_desc.setMatMulTileSizeInBytes(32);
  }

 private:
  DeviceDesc cpu_device_desc;
};

} // namespace mlir
#endif // MLIR_SUPPORT_SYSTEMDESC_H
