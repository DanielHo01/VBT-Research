// Minimal nlohmann/json stub for LSP validation ONLY.
// Real nlohmann/json is linked at build time (header-only).
// This file exists ONLY so clangd can validate vbtcore-cpp structure
// without requiring a full nlohmann/json install. DO NOT compile against this stub.

#pragma once
#include <cstddef>
#include <initializer_list>
#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace nlohmann {

class json {
public:
    using value_type = std::map<std::string, json>;
    using array_type = std::vector<json>;
    using size_type = std::size_t;

    json() : type_(NULL_T), bool_val_(false), int_val_(0), double_val_(0.0) {}
    json(std::nullptr_t) : json() {}
    json(bool v) : type_(BOOL), bool_val_(v), int_val_(0), double_val_(0.0) {}
    json(int v) : type_(INT), bool_val_(false), int_val_(v), double_val_(0.0) {}
    json(double v) : type_(DOUBLE), bool_val_(false), int_val_(0), double_val_(v) {}
    json(const char* v) : type_(STRING), bool_val_(false), int_val_(0),
                          double_val_(0.0), str_val_(v ? v : "") {}
    json(const std::string& v) : type_(STRING), bool_val_(false), int_val_(0),
                                  double_val_(0.0), str_val_(v) {}
    json(const array_type& v) : type_(ARRAY), bool_val_(false), int_val_(0),
                                double_val_(0.0), arr_val_(v) {}
    json(const value_type& v) : type_(OBJECT), bool_val_(false), int_val_(0),
                                double_val_(0.0), obj_val_(v) {}

    static json array() { return json(array_type{}); }
    static json object() { return json(value_type{}); }

    template <typename T>
    static json parse(const T& /*s*/) { return json(); }

    template <typename T>
    T value(const std::string& key, const T& default_val) const {
        (void)key;
        return default_val;
    }
    std::string value(const std::string& key, const char* default_val) const {
        (void)key;
        return std::string(default_val ? default_val : "");
    }

    bool contains(const std::string& key) const { (void)key; return false; }
    bool is_string() const { return type_ == STRING; }
    bool is_array() const { return type_ == ARRAY; }
    bool is_object() const { return type_ == OBJECT; }

    // operator[] — 唯一重载（const char*），所有字符串键调用经此路径
    // 避免 const char[N] / const std::string& / size_type 多重重载歧义
    json& operator[](const char* key) {
        if (type_ != OBJECT) { type_ = OBJECT; obj_val_.clear(); }
        return obj_val_[std::string(key ? key : "")];
    }
    const json& operator[](const char* key) const {
        static json j;
        (void)key;
        return j;
    }

    size_type size() const {
        return type_ == ARRAY ? arr_val_.size()
             : type_ == OBJECT ? obj_val_.size()
             : 0;
    }

    void push_back(const json& v) {
        if (type_ != ARRAY) { type_ = ARRAY; arr_val_.clear(); }
        arr_val_.push_back(v);
    }

    json(std::initializer_list<std::pair<const std::string, json>> init) {
        type_ = OBJECT;
        bool_val_ = false;
        int_val_ = 0;
        double_val_ = 0.0;
        for (auto& kv : init) obj_val_[kv.first] = kv.second;
    }
    json(std::initializer_list<json> init) {
        type_ = ARRAY;
        bool_val_ = false;
        int_val_ = 0;
        double_val_ = 0.0;
        for (const auto& v : init) arr_val_.push_back(v);
    }

    friend std::ostream& operator<<(std::ostream& os, const json& /*j*/) {
        return os << "{}";
    }

    std::string dump(int /*indent*/ = -1) const { return "{}"; }

private:
    enum Type { NULL_T, BOOL, INT, DOUBLE, STRING, ARRAY, OBJECT };
    Type type_;
    bool bool_val_;
    int int_val_;
    double double_val_;
    std::string str_val_;
    array_type arr_val_;
    value_type obj_val_;
};

}  // namespace nlohmann