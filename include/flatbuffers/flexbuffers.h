/*
 * Copyright 2015 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLATBUFFERS_FLEXBUFFERS_H_
#define FLATBUFFERS_FLEXBUFFERS_H_

// We use the basic binary writing functions from the regular FlatBuffers.
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/util.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace flexbuffers {

class Reference;
class Map;

enum BitWidth : uint8_t {
  BIT_WIDTH_8 = 0,
  BIT_WIDTH_16 = 1,
  BIT_WIDTH_32 = 2,
  BIT_WIDTH_64 = 3,
};

enum Type : uint8_t {
  SL_NULL = 0,
  SL_INT = 1,
  SL_UINT = 2,
  SL_FLOAT = 3,
  // Types above stored inline, types below over an offset.
  SL_KEY = 4,
  SL_STRING = 5,
  SL_INDIRECT_INT = 6,
  SL_INDIRECT_UINT = 7,
  SL_INDIRECT_FLOAT = 8,
  SL_MAP = 9,
  SL_VECTOR = 10,        // Untyped.
  SL_VECTOR_INT = 11,    // Typed any size.
  SL_VECTOR_UINT = 12,
  SL_VECTOR_FLOAT = 13,
  SL_VECTOR_KEY = 14,
  SL_VECTOR_INT2 = 15,   // Typed tuple (no size field).
  SL_VECTOR_UINT2 = 16,
  SL_VECTOR_FLOAT2 = 17,
  SL_VECTOR_INT3 = 18,   // Typed triple (no size field).
  SL_VECTOR_UINT3 = 19,
  SL_VECTOR_FLOAT3 = 20,
  SL_VECTOR_INT4 = 21,   // Typed quad (no size field).
  SL_VECTOR_UINT4 = 22,
  SL_VECTOR_FLOAT4 = 23,
  SL_BLOB = 24,
};

inline bool IsInline(Type t) { return t <= SL_FLOAT; }

inline bool IsTypedVectorElementType(Type t) {
  return t >= SL_INT && t <= SL_KEY;
}

inline bool IsTypedVector(Type t) {
  return t >= SL_VECTOR_INT && t <= SL_VECTOR_KEY;
}

inline Type ToTypedVector(Type t, int fixed_len = 0) {
  switch (fixed_len) {
    case 0: return static_cast<Type>(t - SL_INT + SL_VECTOR_INT);
    case 2: return static_cast<Type>(t - SL_INT + SL_VECTOR_INT2);
    case 3: return static_cast<Type>(t - SL_INT + SL_VECTOR_INT3);
    case 4: return static_cast<Type>(t - SL_INT + SL_VECTOR_INT4);
    default: assert(0); return SL_NULL;
  }
}

inline Type ToTypedVectorElementType(Type t) {
  return static_cast<Type>(t - SL_VECTOR_INT + SL_INT);
}

inline bool IsFixedTypedVector(Type t) {
  return t >= SL_VECTOR_INT2 && t <= SL_VECTOR_FLOAT4;
}

inline Type ToFixedTypedVectorElementType(Type t, uint8_t *len) {
  auto fixed_type = t - SL_VECTOR_INT2;
  *len = fixed_type / 3 + 2;  // 3 types each, starting from length 2.
  return static_cast<Type>(fixed_type % 3 + SL_INT);
}

// TODO: implement proper support for 8/16bit floats, or decide not to
// support them.
typedef int16_t half;
typedef int8_t quarter;


// TODO: can we do this without conditionals using intrinsics or inline asm
// on some platforms? Given branch prediction the method below should be
// decently quick, but it is the most frequently executed function.

template <typename R, typename T1, typename T2, typename T4, typename T8>
R ReadSizedScalar(const uint8_t *data, uint8_t byte_width) {
  return byte_width < 4
    ? (byte_width < 2 ? static_cast<R>(flatbuffers::ReadScalar<T1>(data))
                      : static_cast<R>(flatbuffers::ReadScalar<T2>(data)))
    : (byte_width < 8 ? static_cast<R>(flatbuffers::ReadScalar<T4>(data))
                      : static_cast<R>(flatbuffers::ReadScalar<T8>(data)));
}


inline int64_t ReadInt64(const uint8_t *data, uint8_t byte_width) {
  return ReadSizedScalar<int64_t, int8_t, int16_t, int32_t, int64_t>(data,
           byte_width);
}

inline uint64_t ReadUInt64(const uint8_t *data, uint8_t byte_width) {
  // This is the "hottest" function (all offset lookups use this), so worth
  // optimizing if possible.
  // TODO: GCC apparently replaces memcpy by a rep movsb, but only if count is a
  // constant, which here it isn't. Test if memcpy is still faster than
  // the conditionals in ReadSizedScalar. Can also use inline asm.
  #ifdef _MSC_VER
    uint64_t u = 0;
    __movsb(reinterpret_cast<int8_t *>(&u),
            reinterpret_cast<const int8_t *>(data), byte_width);
    return flatbuffers::EndianScalar(u);
  #else
    return ReadSizedScalar<uint64_t, uint8_t, uint16_t, uint32_t, uint64_t>(
             data, byte_width);
  #endif
}

inline double ReadDouble(const uint8_t *data, uint8_t byte_width) {
  return ReadSizedScalar<double, quarter, half, float, double>(data,
           byte_width);
}

const uint8_t *Indirect(const uint8_t *offset, uint8_t byte_width) {
  return offset - ReadUInt64(offset, byte_width);
}

template<typename T> const uint8_t *Indirect(const uint8_t *offset) {
  return offset - flatbuffers::ReadScalar<T>(offset);
}

class Object {
 public:
  Object(const uint8_t *data, uint8_t byte_width)
    : data_(data), byte_width_(byte_width) {}

 protected:
  const uint8_t *data_;
  uint8_t byte_width_;
};

class Sized : public Object {
 public:
  Sized(const uint8_t *data, uint8_t byte_width) : Object(data, byte_width) {}
  size_t size() const {
    return static_cast<size_t>(ReadUInt64(data_ - byte_width_, byte_width_));
  }
};

class String : public Sized {
 public:
  String(const uint8_t *data, uint8_t byte_width)
    : Sized(data, byte_width) {}

  size_t length() const { return size(); }
  const char *c_str() const { return reinterpret_cast<const char *>(data_); }

  static String EmptyString() {
    static const uint8_t empty_string[] = { 0/*len*/, 0/*terminator*/ };
    return String(empty_string + 1, 1);
  }
  bool IsTheEmptyString() const { return data_ == EmptyString().data_; }
};

class Blob : public Sized {
 public:
  Blob(const uint8_t *data, uint8_t byte_width)
    : Sized(data, byte_width) {}

  static Blob EmptyBlob() {
    static const uint8_t empty_blob[] = { 0/*len*/ };
    return Blob(empty_blob + 1, 1);
  }
  bool IsTheEmptyBlob() const { return data_ == EmptyBlob().data_; }
};

class Vector : public Sized {
 public:
  Vector(const uint8_t *data, uint8_t byte_width)
    : Sized(data, byte_width) {}

  Reference operator[](size_t i) const;

  static Vector EmptyVector() {
    static const uint8_t empty_vector[] = { 0/*len*/ };
    return Vector(empty_vector + 1, 1);
  }
  bool IsTheEmptyVector() const { return data_ == EmptyVector().data_; }
};

class TypedVector : public Sized {
 public:
  TypedVector(const uint8_t *data, uint8_t byte_width, Type element_type)
    : Sized(data, byte_width), type_(element_type) {}

  Reference operator[](size_t i) const;

  static TypedVector EmptyTypedVector() {
    static const uint8_t empty_typed_vector[] = { 0/*len*/ };
    return TypedVector(empty_typed_vector + 1, 1, SL_INT);
  }
  bool IsTheEmptyVector() const {
    return data_ == TypedVector::EmptyTypedVector().data_;
  }

  Type ElementType() { return type_; }

 private:
  Type type_;

  friend Map;
};

class FixedTypedVector : public Object {
 public:
  FixedTypedVector(const uint8_t *data, uint8_t byte_width, Type element_type,
                   uint8_t len)
    : Object(data, byte_width), type_(element_type), len_(len) {}

  Reference operator[](size_t i) const;

  static FixedTypedVector EmptyFixedTypedVector() {
    static const uint8_t fixed_empty_vector[] = { 0/* unused */ };
    return FixedTypedVector(fixed_empty_vector, 1, SL_INT, 0);
  }
  bool IsTheEmptyFixedTypedVector() const {
    return data_ == FixedTypedVector::EmptyFixedTypedVector().data_;
  }

  Type ElementType() { return type_; }
  uint8_t size() { return len_; }

 private:
  Type type_;
  uint8_t len_;
};

class Map : public Vector {
 public:
  Map(const uint8_t *data, uint8_t byte_width)
    : Vector(data, byte_width) {}

  Reference operator[](const char *key) const;
  Reference operator[](const std::string &key) const;

  TypedVector Keys() const {
    auto keys_offset = data_ - byte_width_ * 3;
    return TypedVector(Indirect(keys_offset, byte_width_),
                       ReadUInt64(keys_offset + byte_width_, byte_width_),
                       SL_KEY);
  }

  static Map EmptyMap() {
    static const uint8_t empty_map[] = {
      0/*keys_len*/, 1/*keys_width*/, 1/*keys_offset*/, 0/*len*/
    };
    return Map(empty_map + 4, 1);
  }

  bool IsTheEmptyMap() const {
    return data_ == EmptyMap().data_;
  }
};

class Reference {
 public:
  Reference(const uint8_t *data, uint8_t parent_width, uint8_t byte_width,
            Type type)
    : data_(data), parent_width_(parent_width), byte_width_(byte_width),
      type_(type) {}

  Reference(const uint8_t *data, uint8_t parent_width, uint8_t packed_type)
    : data_(data), parent_width_(parent_width) {
    byte_width_ = 1U << static_cast<BitWidth>(packed_type & 3);
    type_ = static_cast<Type>(packed_type >> 2);
  }

  Type GetType() { return type_; }

  bool IsNull() { return type_ == SL_NULL; }
  bool IsInt() { return type_ == SL_INT || type_ == SL_INDIRECT_INT; }
  bool IsUInt() { return type_ == SL_UINT || type_ == SL_INDIRECT_UINT;; }
  bool IsIntOrUint() { return IsInt() || IsUInt(); }
  bool IsFloat() { return type_ == SL_FLOAT || type_ == SL_INDIRECT_FLOAT; }
  bool IsNumeric() { return IsIntOrUint() || IsFloat(); }
  bool IsString() { return type_ == SL_STRING; }
  bool IsKey() { return type_ == SL_KEY; }
  bool IsVector() { return type_ == SL_VECTOR || type_ == SL_MAP; }
  bool IsMap() { return type_ == SL_MAP; }

  // Reads any type as a int64_t. Never fails, does most sensible conversion.
  // Truncates floats, strings are attempted to be parsed for a number,
  // vectors/maps return their size. Returns 0 if all else fails.
  int64_t AsInt64() {
    // Use if instead of switch, since these are ordered from most to least
    // likely to be used, so should on average be faster than switch.
    if (type_ == SL_INT) {
      return ReadInt64(data_, parent_width_);
    } else if (type_ == SL_INDIRECT_INT) {
      return ReadInt64(Indirect(), byte_width_);
    } else if (type_ == SL_UINT) {
      return ReadUInt64(data_, parent_width_);
    } else if (type_ == SL_INDIRECT_UINT) {
      return ReadUInt64(Indirect(), byte_width_);
    } else if (type_ == SL_FLOAT) {
      return static_cast<int64_t>(ReadDouble(data_, parent_width_));
    } else if (type_ == SL_INDIRECT_FLOAT) {
      return static_cast<int64_t>(ReadDouble(Indirect(), byte_width_));
    } else if (type_ == SL_NULL) {
      return 0;
    } else if (type_ == SL_STRING) {
      return flatbuffers::StringToInt(AsString().c_str());
    } else if (type_ == SL_VECTOR) {
      return static_cast<int64_t>(AsVector().size());
    } else {
      // Convert other things to int.
      return 0;
    }
  }

  // TODO: could specialize these to not use AsInt64() if that saves
  // extension ops in generated code, and use a faster op than ReadInt64.
  int32_t AsInt32() { return static_cast<int32_t>(AsInt64()); }
  int16_t AsInt16() { return static_cast<int16_t>(AsInt64()); }
  int8_t  AsInt8()  { return static_cast<int8_t> (AsInt64()); }

  uint64_t AsUInt64() {
    if (type_ == SL_UINT) {
      return ReadUInt64(data_, parent_width_);
    } else if (type_ == SL_INDIRECT_UINT) {
      return ReadUInt64(Indirect(), byte_width_);
    } else if (type_ == SL_INT) {
      return ReadInt64(data_, parent_width_);
    } else if (type_ == SL_INDIRECT_INT) {
      return ReadInt64(Indirect(), byte_width_);
    } else if (type_ == SL_FLOAT) {
      return static_cast<uint64_t>(ReadDouble(data_, parent_width_));
    } else if (type_ == SL_INDIRECT_FLOAT) {
      return static_cast<uint64_t>(ReadDouble(Indirect(), byte_width_));
    } else if (type_ == SL_NULL) {
      return 0;
    } else if (type_ == SL_STRING) {
      return flatbuffers::StringToUInt(AsString().c_str());
    } else if (type_ == SL_VECTOR) {
      return static_cast<uint64_t>(AsVector().size());
    } else {
      // Convert other things to uint.
      return 0;
    }
  }

  uint32_t AsUInt32() { return static_cast<uint32_t>(AsUInt64()); }
  uint16_t AsUInt16() { return static_cast<uint16_t>(AsUInt64()); }
  uint8_t  AsUInt8()  { return static_cast<uint8_t> (AsUInt64()); }

  double AsDouble() {
    if (type_ == SL_FLOAT) {
      return ReadDouble(data_, parent_width_);
    } else if (type_ == SL_INDIRECT_FLOAT) {
      return ReadDouble(Indirect(), byte_width_);
    } else if (type_ == SL_INT) {
      return static_cast<double>(ReadInt64(data_, parent_width_));
    } else if (type_ == SL_UINT) {
      return static_cast<double>(ReadUInt64(data_, parent_width_));
    } else if (type_ == SL_INDIRECT_INT) {
      return static_cast<double>(ReadInt64(Indirect(), byte_width_));
    } else if (type_ == SL_INDIRECT_UINT) {
      return static_cast<double>(ReadUInt64(Indirect(), byte_width_));
    } else if (type_ == SL_NULL) {
      return 0.0;
    } else if (type_ == SL_STRING) {
      return strtod(AsString().c_str(), nullptr);
    } else if (type_ == SL_VECTOR) {
      return static_cast<double>(AsVector().size());
    } else {
      // Convert strings and other things to float.
      return 0;
    }
  }

  float AsFloat() { return static_cast<float>(AsDouble()); }

  const char *AsKey() {
    if (type_ == SL_KEY) {
      return reinterpret_cast<const char *>(Indirect());
    } else {
      return "";
    }
  }

  // This function returns the empty string if you try to read a not-string.
  String AsString() {
    if (type_ == SL_STRING) {
      return String(Indirect(), byte_width_);
    } else {
      return String::EmptyString();
    }
  }

  // Unlike AsString(), this will convert any type to a std::string.
  std::string ToString() {
    if (type_ == SL_STRING) {
      return String(Indirect(), byte_width_).c_str();
    } else if (IsKey()) {
      return AsKey();
    } else if (IsInt()) {
      return flatbuffers::NumToString(AsInt64());
    } else if (IsUInt()) {
      return flatbuffers::NumToString(AsUInt64());
    } else if (IsFloat()) {
      return flatbuffers::NumToString(AsDouble());
    } else if (IsNull()) {
      return "null";
    } else if (IsMap()) {
      return "{..}";  // TODO: show elements.
    } else if (IsVector()) {
      return "[..]";  // TODO: show elements.
    } else {
      return "(?)";
    }
  }

  // This function returns the empty blob if you try to read a not-blob.
  // Strings can be viewed as blobs too.
  Blob AsBlob() {
    if (type_ == SL_BLOB || type_ == SL_STRING) {
      return Blob(Indirect(), byte_width_);
    } else {
      return Blob::EmptyBlob();
    }
  }

  // This function returns the empty vector if you try to read a not-vector.
  // Maps can be viewed as vectors too.
  Vector AsVector() {
    if (type_ == SL_VECTOR || type_ == SL_MAP) {
      return Vector(Indirect(), byte_width_);
    } else {
      return Vector::EmptyVector();
    }
  }

  TypedVector AsTypedVector() {
    if (IsTypedVector(type_)) {
      return TypedVector(Indirect(), byte_width_,
                         ToTypedVectorElementType(type_));
    } else {
      return TypedVector::EmptyTypedVector();
    }
  }

  FixedTypedVector AsFixedTypedVector() {
    if (IsFixedTypedVector(type_)) {
      uint8_t len = 0;
      auto vtype = ToFixedTypedVectorElementType(type_, &len);
      return FixedTypedVector(Indirect(), byte_width_, vtype, len);
    } else {
      return FixedTypedVector::EmptyFixedTypedVector();
    }
  }

  Map AsMap() {
    if (type_ == SL_MAP) {
      return Map(Indirect(), byte_width_);
    } else {
      return Map::EmptyMap();
    }
  }

 private:
  const uint8_t *Indirect() {
    return flexbuffers::Indirect(data_, parent_width_);
  }

  const uint8_t *data_;
  uint8_t parent_width_;
  uint8_t byte_width_;
  Type type_;
};

inline uint8_t PackedType(BitWidth bit_width, Type type) {
  return static_cast<uint8_t>(bit_width | (type << 2));
}

inline uint8_t NullPackedType() {
  return PackedType(BIT_WIDTH_8, SL_NULL);
}

inline Reference Vector::operator[](size_t i) const  {
  auto len = size();
  if (i >= len) return Reference(nullptr, 1, NullPackedType());
  auto packed_type = (data_ + len * byte_width_)[i];
  auto elem = data_ + i * byte_width_;
  return Reference(elem, byte_width_, packed_type);
}

inline Reference TypedVector::operator[](size_t i) const  {
  auto len = size();
  if (i >= len) return Reference(nullptr, 1, NullPackedType());
  auto elem = data_ + i * byte_width_;
  return Reference(elem, byte_width_, 1, type_);
}

inline Reference FixedTypedVector::operator[](size_t i) const  {
  if (i >= len_) return Reference(nullptr, 1, NullPackedType());
  auto elem = data_ + i * byte_width_;
  return Reference(elem, byte_width_, 1, type_);
}

template<typename T> int KeyCompare(const void *key, const void *elem) {
  auto str_elem = reinterpret_cast<const char *>(
                    Indirect<T>(reinterpret_cast<const uint8_t *>(elem)));
  auto skey = reinterpret_cast<const char *>(key);
  return strcmp(skey, str_elem);
}

inline Reference Map::operator[](const char *key) const {
  auto keys = Keys();
  // We can't pass keys.byte_width_ to the comparison function, so we have
  // to pick the right one ahead of time.
  int (*comp)(const void *, const void *) = nullptr;
  switch (keys.byte_width_) {
    case 1: comp = KeyCompare<uint8_t>; break;
    case 2: comp = KeyCompare<uint16_t>; break;
    case 4: comp = KeyCompare<uint32_t>; break;
    case 8: comp = KeyCompare<uint64_t>; break;
  }
  auto res = std::bsearch(key, keys.data_, keys.size(), keys.byte_width_, comp);
  if (!res)
    return Reference(nullptr, 1, NullPackedType());
  auto i = (reinterpret_cast<uint8_t *>(res) - keys.data_) / keys.byte_width_;
  return (*static_cast<const Vector *>(this))[i];
}

inline Reference Map::operator[](const std::string &key) const {
  return (*this)[key.c_str()];
}

inline Reference GetRoot(const uint8_t *buffer, size_t size) {
  auto byte_width = buffer[size - 1];
  auto packed_type = buffer[size - 2];
  return Reference(buffer + size - byte_width - 2, byte_width, packed_type);
}

inline Reference GetRoot(const std::vector<uint8_t> &buffer) {
  return GetRoot(buffer.data(), buffer.size());
}

class Builder FLATBUFFERS_FINAL_CLASS {
 public:
  Builder(size_t initial_size = 256)
      : buf_(initial_size), finished_(false) {
    buf_.clear();
  }

  /// @brief Get the serialized buffer (after you call `Finish()`).
  /// @return Returns a vector owned by this class.
  const std::vector<uint8_t> &GetBuffer() const {
    Finished();
    return buf_;
  }

  // All value constructing functions below have two versions: one that
  // takes a key (for placement inside a map) and one that doesn't (for inside
  // vectors and elsewhere).

  void Null() { stack_.push_back(Value()); }
  void Null(const char *key) { Key(key); Null(); }

  void Int(int64_t i) { stack_.push_back(Value(i, SL_INT, WidthI(i))); }
  void Int(const char *key, int64_t i) { Key(key); Int(i); }

  void UInt(uint64_t u) { stack_.push_back(Value(u, SL_UINT, WidthU(u))); }
  void UInt(const char *key, uint64_t u) { Key(key); Int(u); }

  void Float(float f) { stack_.push_back(Value(f)); }
  void Float(const char *key, float f) { Key(key); Float(f); }

  void Double(double f) { stack_.push_back(Value(f)); }
  void Double(const char *key, double d) { Key(key); Double(d); }

  void Bool(bool b) { Int(static_cast<int64_t>(b)); }
  void Bool(const char *key, bool b) { Key(key); Bool(b); }

  void IndirectInt(int64_t i) {
    PushIndirect(i, SL_INDIRECT_INT, WidthI(i));
  }
  void IndirectInt(const char *key, int64_t i) {
    Key(key);
    IndirectInt(i);
  }

  void IndirectUInt(uint64_t u) {
    PushIndirect(u, SL_INDIRECT_UINT, WidthU(u));
  }
  void IndirectUInt(const char *key, uint64_t u) {
    Key(key);
    IndirectUInt(u);
  }

  void IndirectFloat(float f) {
    PushIndirect(f, SL_INDIRECT_FLOAT, BIT_WIDTH_32);
  }
  void IndirectFloat(const char *key, float f) {
    Key(key);
    IndirectFloat(f);
  }

  void IndirectDouble(double f) {
    PushIndirect(f, SL_INDIRECT_FLOAT, BIT_WIDTH_64);
  }
  void IndirectDouble(const char *key, double d) {
    Key(key);
    IndirectDouble(d);
  }

  size_t Key(const char *str, size_t len) {
    // TODO: add auto string pooling if requested.
    auto sloc = buf_.size();
    WriteBytes(str, len + 1);
    stack_.push_back(Value(static_cast<uint64_t>(sloc), SL_KEY, BIT_WIDTH_8));
    return sloc;
  }

  size_t Key(const char *str) { return Key(str, strlen(str)); }
  size_t Key(const std::string &str) { return Key(str.c_str(), str.size()); }

  size_t String(const char *str, size_t len) {
    // TODO: add auto string pooling if requested.
    return CreateBlob(str, len, 1, SL_STRING);
  }
  size_t String(const char *str) {
    return String(str, strlen(str));
  }
  size_t String(const std::string &str) {
    return String(str.c_str(), str.size());
  }
  void String(const flexbuffers::String &str) {
    String(str.c_str(), str.length());
  }

  void String(const char *key, const char *str) {
    Key(key);
    String(str);
  }
  void String(const char *key, const std::string &str) {
    Key(key);
    String(str);
  }
  void String(const char *key, const flexbuffers::String &str) {
    Key(key);
    String(str);
  }

  size_t Blob(const void *data, size_t len) {
    return CreateBlob(data, len, 0, SL_BLOB);
  }
  size_t Blob(const std::vector<uint8_t> &v) {
    return CreateBlob(v.data(), v.size(), 0, SL_BLOB);
  }

  // TODO(wvo): support all the FlexBuffer types (like flexbuffers::String),
  // e.g. Vector etc. Also in overloaded versions.
  // Also some FlatBuffers types?

  size_t StartVector() { return stack_.size(); }
  size_t StartVector(const char *key) { Key(key); return stack_.size(); }
  size_t StartMap() { return stack_.size(); }
  size_t StartMap(const char *key) { Key(key); return stack_.size(); }

  size_t EndVector(size_t start, bool typed, bool fixed) {
    auto vec = CreateVector(start, stack_.size() - start, 1, typed, fixed);
    // Remove temp elements and return vector.
    stack_.resize(start);
    stack_.push_back(vec);
    return vec.u_;
  }

  size_t EndMap(size_t start) {
    // We should have interleaved keys and values on the stack.
    // Make sure it is an even number:
    auto len = stack_.size() - start;
    assert(!(len & 1));
    len /= 2;
    // Make sure keys are all strings:
    for (auto key = start; key < stack_.size(); key += 2) {
      assert(stack_[key].type_ == SL_KEY);
    }
    // Now sort values, so later we can do a binary seach lookup.
    // We want to sort 2 array elements at a time.
    struct TwoValue { Value key; Value val; };
    // TODO: strict aliasing?
    auto dict = reinterpret_cast<TwoValue *>(stack_.data() + start);
    std::sort(dict, dict + len, [&](const TwoValue &a, const TwoValue &b) {
      auto as = reinterpret_cast<const char *>(buf_.data() + a.key.u_);
      auto bs = reinterpret_cast<const char *>(buf_.data() + b.key.u_);
      return strcmp(as, bs) < 0;
    });
    // First create a vector out of all keys.
    // FIXME: make this a typed vector of string.
    auto keys = CreateVector(start, len, 2, true, false);
    auto vec = CreateVector(start + 1, len, 2, false, false,
                            buf_.size() - keys.u_,
                            1U << keys.min_bit_width_);
    // Remove temp elements and return map.
    stack_.resize(start);
    stack_.push_back(vec);
    return vec.u_;
  }

  template<typename F> size_t Vector(F f) {
    auto start = StartVector();
    f();
    return EndVector(start, false, false);
  }
  template<typename F> size_t Vector(const char *key, F f) {
    auto start = StartVector(key);
    f();
    return EndVector(start, false, false);
  }

  template<typename F> size_t TypedVector(F f) {
    auto start = StartVector();
    f();
    return EndVector(start, true, false);
  }
  template<typename F> size_t TypedVector(const char *key, F f) {
    auto start = StartVector(key);
    f();
    return EndVector(start, true, false);
  }

  template<typename T> size_t FixedTypedVector(const T *elems, size_t len) {
    // We only support a few fixed vector lengths. Anything bigger use a
    // regular typed vector.
    assert(len >= 2 && len <= 4);
    // And only scalar values.
    assert(std::is_scalar<T>::value);
    auto start = StartVector();
    for (size_t i = 0; i < len; i++) Add(elems[i]);
    return EndVector(start, true, true);
  }

  template<typename T> size_t FixedTypedVector(const char *key, const T *elems,
                                               size_t len) {
    Key(key);
    return FixedTypedVector(elems, len);
  }

  template<typename F> size_t Map(F f) {
    auto start = StartMap();
    f();
    return EndMap(start);
  }
  template<typename F> size_t Map(const char *key, F f) {
    auto start = StartMap(key);
    f();
    return EndMap(start);
  }

  // Overloaded Add that tries to call the correct function above.
  void Add(int8_t i) { Int(i); }
  void Add(int16_t i) { Int(i); }
  void Add(int32_t i) { Int(i); }
  void Add(int64_t i) { Int(i); }
  void Add(uint8_t u) { UInt(u); }
  void Add(uint16_t u) { UInt(u); }
  void Add(uint32_t u) { UInt(u); }
  void Add(uint64_t u) { UInt(u); }
  void Add(float f) { Float(f); }
  void Add(double d) { Double(d); }
  void Add(bool b) { Bool(b); }
  void Add(const char *str) { String(str); }
  void Add(const std::string &str) { String(str); }
  void Add(const flexbuffers::String &str) { String(str); }

  template<typename T> void Add(const std::vector<T> &vec) {
    auto start = StartVector();
    for (auto it = vec.begin(); it != vec.end(); ++it) Add(*it);
    EndVector(start, true, false);
  }

  template<typename T> void Add(const char *key, T t) {
    Key(key);
    Add(t);
  }

  template<typename T> void Add(const std::map<std::string, T> &map) {
    auto start = StartMap();
    for (auto it = map.begin(); it != map.end(); ++it)
      Add(it->first.c_str(), it->second);
    EndMap(start);
  }

  void Finish() {
    // If you hit this assert, you likely have objects that were never included
    // in a parent. You need to have exactly one root to finish a buffer.
    // Check your Start/End calls are matched, and all objects are inside
    // some other object.
    assert(stack_.size() == 1);

    // Write root value.
    auto byte_width = Align(stack_[0].ElemWidth(buf_.size(), 0));
    WriteAny(stack_[0], byte_width);
    // Write root type.
    Write(stack_[0].StoredPackedType(), 1);
    // Write root size. Normally determined by parent, but root has no parent :)
    Write(byte_width, 1);

    finished_ = true;
  }

 private:
  void Finished() const {
    // If you get this assert, you're attempting to get access a buffer
    // which hasn't been finished yet. Be sure to call
    // Builder::Finish with your root object.
    assert(finished_);
  }

  // Align to prepare for writing a scalar with a certain size.
  uint8_t Align(BitWidth alignment) {
    auto byte_width = 1U << alignment;
    buf_.insert(buf_.end(), flatbuffers::PaddingBytes(buf_.size(), byte_width),
                0);
    return byte_width;
  }

  void WriteBytes(const void *val, size_t size) {
    buf_.insert(buf_.end(),
                reinterpret_cast<const uint8_t *>(val),
                reinterpret_cast<const uint8_t *>(val) + size);
  }

  // For values T >= byte_width
  template<typename T> void Write(T val, uint8_t byte_width) {
    val = flatbuffers::EndianScalar(val);
    WriteBytes(&val, byte_width);
  }

  void WriteDouble(double f, uint8_t byte_width) {
    switch (byte_width) {
      case 8: Write(f, byte_width); break;
      case 4: Write(static_cast<float>(f), byte_width); break;
      //case 2: Write(static_cast<half>(f), byte_width); break;
      //case 1: Write(static_cast<quarter>(f), byte_width); break;
      default: assert(0);
    }
  }

  void WriteOffset(uint64_t o, uint8_t byte_width) {
    auto reloff = buf_.size() - o;
    assert(reloff < 1ULL << (byte_width * 8) || byte_width == 8);
    Write(reloff, byte_width);
  }

  template<typename T> void PushIndirect(T val, Type type, BitWidth bit_width) {
    auto byte_width = Align(bit_width);
    auto iloc = buf_.size();
    Write(val, byte_width);
    stack_.push_back(Value(static_cast<uint64_t>(iloc), type, bit_width));
  }

  static BitWidth WidthU(uint64_t u) {
    if (!(u & 0xFFFFFFFFFFFFFF00)) return BIT_WIDTH_8;
    if (!(u & 0xFFFFFFFFFFFF0000)) return BIT_WIDTH_16;
    if (!(u & 0xFFFFFFFF00000000)) return BIT_WIDTH_32;
    return BIT_WIDTH_64;
  }

  static BitWidth WidthI(int64_t i) {
    auto u = static_cast<uint64_t>(i) << 1;
    return WidthU(i >= 0 ? u : ~u);
  }

  struct Value {
    union {
      int64_t i_;
      uint64_t u_;
      double f_;
    };

    Type type_;

    // For scalars: of itself, for vector: of its elements, for string: length.
    BitWidth min_bit_width_;

    Value() : i_(0), type_(SL_NULL), min_bit_width_(BIT_WIDTH_8) {}

    Value(int64_t i, Type t, BitWidth bw)
      : i_(i), type_(t), min_bit_width_(bw) {}
    Value(uint64_t u, Type t, BitWidth bw)
      : u_(u), type_(t), min_bit_width_(bw) {}

    Value(float f)
      : f_(f), type_(SL_FLOAT), min_bit_width_(BIT_WIDTH_32) {}
    Value(double f)
      : f_(f), type_(SL_FLOAT), min_bit_width_(BIT_WIDTH_64) {}

    uint8_t StoredPackedType(BitWidth parent_bit_width_= BIT_WIDTH_8) {
      return PackedType(StoredWidth(parent_bit_width_), type_);
    }

    BitWidth ElemWidth(size_t buf_size, size_t elem_index) {
      if (IsInline(type_)) {
        return min_bit_width_;
      } else {
        // We have an absolute offset, but want to store a relative offset
        // elem_index elements beyond the current buffer end. Since whether
        // the relative offset fits in a certain byte_width depends on
        // the size of the elements before it (and their alignment), we have
        // to test for each size in turn.
        for (size_t byte_width = 1;
             byte_width <= sizeof(flatbuffers::largest_scalar_t);
             byte_width *= 2) {
          // Where are we going to write this offset?
          auto offset_loc =
            buf_size +
            flatbuffers::PaddingBytes(buf_size, byte_width) +
            elem_index * byte_width;
          // Compute relative offset.
          auto offset = offset_loc - u_;
          // Does it fit?
          auto bit_width = Builder::WidthU(offset);
          if (1U << bit_width == byte_width) return bit_width;
        }
        assert(false);  // Must match one of the sizes above.
        return BIT_WIDTH_64;
      }
    }

    BitWidth StoredWidth(BitWidth parent_bit_width_ = BIT_WIDTH_8) {
      if (IsInline(type_)) {
          return std::max(min_bit_width_, parent_bit_width_);
      } else {
          return min_bit_width_;
      }
    }
  };

  void WriteAny(const Value &val, uint8_t byte_width) {
    switch (val.type_) {
      case SL_NULL:
      case SL_INT:
        Write(val.i_, byte_width);
        break;
      case SL_UINT:
        Write(val.u_, byte_width);
        break;
      case SL_FLOAT:
        WriteDouble(val.f_, byte_width);
        break;
      default:
        WriteOffset(val.u_, byte_width);
        break;
    }
  }

  size_t CreateBlob(const void *data, size_t len, size_t trailing, Type type) {
    auto bit_width = WidthU(len);
    auto byte_width = Align(bit_width);
    Write<uint64_t>(len, byte_width);
    auto sloc = buf_.size();
    WriteBytes(data, len + trailing);
    stack_.push_back(Value(static_cast<uint64_t>(sloc), type, bit_width));
    return sloc;
  }

  Value CreateVector(size_t start, size_t vec_len, size_t step, bool typed,
                     bool fixed,
                     uint64_t keys_rel_offset = 0,
                     uint64_t keys_byte_width = 0) {
    // Figure out smallest bit width we can store this vector with.
    auto bit_width = WidthU(vec_len);
    auto prefix_elems = 1;
    if (keys_rel_offset) {
      // If this vector is part of a map, we will pre-fix an offset to the keys
      // to this vector.
      bit_width = std::max(bit_width, WidthU(keys_rel_offset));
      prefix_elems += 2;
    }
    Type vector_type = SL_KEY;
    // Check bit widths and types for all elements.
    for (size_t i = start; i < stack_.size(); i += step) {
      auto elem_width = stack_[i].ElemWidth(buf_.size(), i + prefix_elems);
      bit_width = std::max(bit_width, elem_width);
      if (typed) {
        if (i == start) {
          vector_type = stack_[i].type_;
        } else {
          // If you get this assert, you are writing a typed vector with
          // elements that are not all the same type.
          assert(vector_type == stack_[i].type_);
        }
      }
    }
    // If you get this assert, your fixed types are not one of:
    // Int / UInt / Float / Key.
    assert(IsTypedVectorElementType(vector_type));
    auto byte_width = Align(bit_width);
    // Write vector. First the keys width/offset if available, and size.
    if (keys_rel_offset) {
      Write(keys_rel_offset, byte_width);
      Write(keys_byte_width, byte_width);
    }
    if (!fixed) {
      Write(vec_len, byte_width);
    }
    // Then the actual data.
    auto vloc = buf_.size();
    for (size_t i = start; i < stack_.size(); i += step) {
      WriteAny(stack_[i], byte_width);
    }
    // Then the types.
    if (!typed) {
      for (size_t i = start; i < stack_.size(); i += step) {
        buf_.push_back(stack_[i].StoredPackedType(bit_width));
      }
    }
    return Value(vloc, keys_rel_offset
                         ? SL_MAP
                         : (typed
                            ? ToTypedVector(vector_type, fixed ? vec_len : 0)
                            : SL_VECTOR),
                       bit_width);
  }

  // You shouldn't really be copying instances of this class.
  Builder(const Builder &);
  Builder &operator=(const Builder &);

  std::vector<uint8_t> buf_;
  std::vector<Value> stack_;

  bool finished_;
};

}  // namespace flexbuffers

#endif  // FLATBUFFERS_FLEXBUFFERS_H_
