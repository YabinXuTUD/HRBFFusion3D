
#ifndef CORE_UTIL_STRINGUTILCONVERT_H_
#define CORE_UTIL_STRINGUTILCONVERT_H_

#include <string>

/////////////////////////////////////////////////////////////////////
// util is already used before; THIS must be included after  //
// all types have been declared for proper conversion              //
/////////////////////////////////////////////////////////////////////

//////////////////////
// native functions //
//////////////////////

namespace convert {
        inline int toInt(const std::string& s) {
                return std::stoi(s);
        }
    inline long long toInt64(const std::string& s) {
        return std::strtoll(s.c_str(), NULL, 10);
    }
    inline int toInt(bool b) {
        if(b) return 1;
        return 0;
    }
        inline unsigned int toUInt(const std::string& s) {
                //return (unsigned int)toInt(s);
                return std::stoul(s.c_str(), NULL, 10);
        }
        inline long long toUInt64(const std::string& s) {
                return std::strtoull(s.c_str(), NULL, 10);
        }
        inline double toDouble(const std::string& s) {
                return std::stod(s);
        }
        inline float toFloat(const std::string& s) {
                return std::stof(s);
        }
        inline char toChar(const std::string& s) {
                return s[0];
        }
        inline bool toBool(const std::string& s) {
                if (s == "false" || s == "False" || s == "0") { return false; }
                else { return true; }
        }
    template<class T>
    inline std::string toString(const std::vector<T>& val) {
        std::string result;
        for (const auto &e : val)
        {
            result = result + std::to_string(e) + " ";
        }
        return result;
    }
        template<class T>
        inline std::string toString(const T& val) {
                return std::to_string(val);
        }


        template<class T> inline void to(const std::string& s, T& res);

        template<>  inline void to<int>(const std::string& s, int& res) {
                res = toInt(s);
        }
        template<>  inline void to<unsigned int>(const std::string& s, unsigned int& res) {
                res = toUInt(s);
        }
        template<>  inline void to<long long>(const std::string& s, long long& res) {
                res = toInt64(s);
        }
        template<>  inline void to<unsigned long long>(const std::string& s, unsigned long long& res) {
                res = toUInt64(s);
        }
        template<>  inline void to<double>(const std::string& s, double& res) {
                res = toDouble(s);
        }
        template<>  inline void to<float>(const std::string& s, float& res) {
                res = toFloat(s);
        }
        template<>  inline void to<std::string>(const std::string& s, std::string& res) {
                res = s;
        }
        template<>  inline void to<char>(const std::string& s, char& res) {
                res = toChar(s);
        }
        template<> inline void to<bool>(const std::string& s, bool& res) {
                res = toBool(s);
        }

}  // namespace Convert

namespace util {

	////////////////////////
	// template overloads //
	////////////////////////
	template<class T> inline T convertTo(const std::string& s) {
		T res;
		convert::to(s, res);
		return res;
	}

	template<class T> inline void convertTo(const std::string& s, T& res) {
		convert::to<T>(s, res);
	}
}  // namespace util

//! stringstream functionality
template<class T>
inline std::string& operator<<(std::string& s, const T& in) {
	s += std::to_string(in);
	return s;
}

#endif  // CORE_UTIL_STRINGUTILCONVERT_H_
