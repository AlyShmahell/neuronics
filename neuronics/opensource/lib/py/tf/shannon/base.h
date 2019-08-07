#include <iostream>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdint.h>
#include <string>
#include "format_id.h"

template <typename T>
std::string join(T a, T b)
{
    char buffer1[32];
    char buffer2[32];
    snprintf(buffer1, sizeof(buffer1), format_id<T>::Get(), a);
    snprintf(buffer2, sizeof(buffer2), format_id<T>::Get(), b);
    return std::string(buffer1) + "|" + std::string(buffer2);
}

template <typename T>
long double entropy(std::map<T, long double> frequencies)
{
    long double result = 0.0;
    for (auto frequency : frequencies)
    {
        result += -frequency.second * log2(frequency.second);
    }
    return result;
}

template <typename T>
double dxentropy(std::map<T, double> frequencies)
{
    double result = 0.0;
    for (auto frequency : frequencies)
    {
        result += -((log(frequency.second) + 1)/log(2));
    }
    return result;
}
