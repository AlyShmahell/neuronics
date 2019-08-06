#include <tensorflow/core/framework/op.h>
#include <stdint.h>
#include <string>

template <typename T>
struct format_id
{
    static const char* Get()
    {
        return typeid(T).name();
    }
};

template <>
struct format_id <tensorflow::int32>
{
    static const char* Get()
    {
        return "%d";
    }
};

template <>
struct format_id <tensorflow::int64>
{
    static const char* Get()
    {
        return "%ld";
    }
};

template <>
struct format_id <float>
{
    static const char* Get()
    {
        return "%f";
    }
};

template <>
struct format_id <double>
{
    static const char* Get()
    {
        return "%g";
    }
};

template <>
struct format_id <std::string>
{
    static const char* Get()
    {
        return "%s";
    }
};
