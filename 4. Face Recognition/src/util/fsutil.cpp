#include "fsutil.h"

#include <sys/types.h>
#include <sys/stat.h>
#ifdef __linux__
#include <unistd.h>
#include <dirent.h>
#elif _WIN32
#include <windows.h>
#endif
namespace fs {

bool pathExists(const std::string& path)
{
	struct stat buf;
	int result = stat(path.c_str(), &buf);
	return !result;
}

bool isDir(const std::string& path)
{
	struct stat buf;
	int result = stat(path.c_str(), &buf);
	if (result)
		return false;
#ifdef __linux__
	return S_ISDIR(buf.st_mode);
#elif _WIN32
	return !!(buf.st_mode & _S_IFDIR);
#endif
}

std::vector<std::string> getFilesInDir(const std::string& path)
{
    std::vector<std::string> result;
#ifdef __linux__
	DIR* pDir = opendir(path.c_str());
	if (!pDir)
		return result;
	dirent* pDirent = nullptr;
	while((pDirent = readdir(pDir))) {
		if (std::string(".") == pDirent->d_name || std::string("..") == pDirent->d_name)
			continue;
		result.push_back(pDirent->d_name);
	}
#elif _WIN32
	WIN32_FIND_DATA findData;
	std::string findPath = fs::concatPath(path, '*');
	HANDLE findHandle = FindFirstFile(findPath.c_str(), &findData);
	if (findHandle == INVALID_HANDLE_VALUE)
		return result;
	do
	{
		if (std::string(".") == findData.cFileName || std::string("..") == findData.cFileName)
			continue;
		result.push_back(findData.cFileName);
	} while (FindNextFile(findHandle, &findData) != 0);
	FindClose(findHandle);
#endif
	return result;
}

std::string concatPathImpl(std::string path, std::string part)
{
	if (path.empty())
		return part;
	bool pathEndsWithSlash = path.back() == '/';
	bool partBeginsWithSlash = part.front() == '/';
	if (pathEndsWithSlash && partBeginsWithSlash)
		path.erase(path.end() - 1);
	else if (!pathEndsWithSlash && !partBeginsWithSlash)
		path += '/';
	path += part;
	return path;
}

std::string concatPath(std::initializer_list<std::string> parts)
{
	std::string composed;
	for (const std::string& s: parts)
		composed = concatPathImpl(composed, s);
	return composed;
}

}
