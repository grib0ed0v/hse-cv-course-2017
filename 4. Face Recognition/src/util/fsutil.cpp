#include "fsutil.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

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
	return S_ISDIR(buf.st_mode);
}

std::vector<std::string> getFilesInDir(const std::string& path)
{
	std::vector<std::string> result;
	DIR* pDir = opendir(path.c_str());
	if (!pDir)
		return result;
	dirent* pDirent = nullptr;
	while((pDirent = readdir(pDir))) {
		if (std::string(".") == pDirent->d_name || std::string("..") == pDirent->d_name)
			continue;
		result.push_back(pDirent->d_name);
	}
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
