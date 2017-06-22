#pragma once
#include <string>
#include <vector>
#include <initializer_list>
#include <sstream>

namespace fs
{
	bool pathExists(const std::string& path);
	bool isDir(const std::string& path);
	std::vector<std::string> getFilesInDir(const std::string& path);
	bool mkdir(const std::string path);

	template<typename T>
	std::string stringify(T value)
	{
		std::ostringstream oss;
		oss << value;
		return oss.str();
	}

	std::string concatPath(std::initializer_list<std::string> parts);

	template<typename... Args>
	std::string concatPath(const Args&... args)
	{
		return concatPath({stringify(args)...});
	}

}
