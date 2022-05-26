#include "cudars/mpeg7_io.h"

namespace mpeg7io {
    // ----------------------------------------------
    // I/O OPERATIONS
    // ----------------------------------------------

    void glob(const std::string globPath, std::vector<std::string>& matchingFiles) {
        glob_t glob_result;
        matchingFiles.clear();

        // glob struct resides on the stack
        memset(&glob_result, 0, sizeof (glob_result));

        ::glob(globPath.c_str(), GLOB_TILDE, NULL, &glob_result);
        for (unsigned int i = 0; i < glob_result.gl_pathc; ++i) {
            matchingFiles.push_back(std::string(glob_result.gl_pathv[i]));
        }
        globfree(&glob_result);
    }

    void getDirectoryFiles(const std::string& dirPath, std::vector<std::string>& matchingFiles) {
        std::cout << "Looking in directory \"" << dirPath << "\"" << std::endl;
        expfs::directory_iterator end_itr; // Default ctor yields past-the-end
        for (expfs::directory_iterator i(dirPath); i != end_itr; ++i) {
            // Skip if not a file
            if (expfs::is_regular_file(i->status())) {
                matchingFiles.push_back(i->path().string());
            }
        }
        std::sort(matchingFiles.begin(), matchingFiles.end());
    }

    std::string generateStampedString(const std::string prefix, const std::string postfix) {
        boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
        std::ostringstream formatter;
        std::string formatstring = prefix + "%Y%m%d_%H%M_%S" + postfix;
        formatter.imbue(std::locale(std::cout.getloc(), new boost::posix_time::time_facet(formatstring.c_str())));
        formatter << now;
        return formatter.str();
    }

    void findComparisonPair(const std::vector<std::string>& inputFilenames, std::vector<std::pair<int, int> >& comPairs) {
        std::string prefix;
        int idx1, idx2;

        idx1 = 0;
        while (idx1 < inputFilenames.size()) {
            // Finds the prefix of inputFilenames[idx1] and finds adjacent filenames 
            // with the same prefix 
            prefix = getPrefix(inputFilenames[idx1]);
            idx2 = idx1 + 1;
            while (idx2 < inputFilenames.size() && getPrefix(inputFilenames[idx2]) == prefix) {
                idx2++;
            }
            // Computes all index pairs
            //        std::cout << "Group \"" << prefix << "\" with " << (idx2 - idx1) << " items: ";
            for (int i1 = idx1; i1 < idx2; ++i1) {
                //            std::cout << "\"" << getShortName(inputFilenames[i1]) << "\" [" << i1 << "], ";
                for (int i2 = i1 + 1; i2 < idx2; ++i2) {
                    comPairs.push_back(std::make_pair(i1, i2));
                }
            }
            //        std::cout << "\n";
            idx1 = idx2;
        }
    }

    // Reads outputFilename for the list of already processed files

    void filterComparisonPair(std::string resumeFilename, std::ostream& outputFile,
            const std::vector<std::string>& inputFilenames, std::vector<std::pair<int, int> >& inputPairs,
            std::vector<std::pair<int, int> >& outputPairs) {
        std::unordered_multimap<std::string, int> indicesMap;
        std::vector<std::pair<int, int> > visitedPairs;
        std::string filenameShort, line, label1, label2;
        int numIn1, numOccl1, numRand1, i1, i2;

        outputPairs.clear();
        // Visits all the lines/items of the output file
        for (int i = 0; i < inputFilenames.size(); ++i) {
            filenameShort = getShortName(inputFilenames[i]);
            indicesMap.insert(std::make_pair(filenameShort, i));
        }

        // Finds all the pairs already visited
        std::ifstream resumeFile(resumeFilename.c_str());
        if (!resumeFile) {
            std::cerr << "Cannot open file \"" << resumeFilename << "\": nothing to resume" << std::endl;
            outputPairs.insert(outputPairs.begin(), inputPairs.begin(), inputPairs.end());
            return;
        }
        while (!resumeFile.eof()) {
            std::getline(resumeFile, line);
            outputFile << line << "\n";
            // Strips comment from line
            size_t pos = line.find_first_of('#');
            if (pos != std::string::npos) {
                line = line.substr(0, pos);
            }
            // Reads the labels of the two files from items 
            std::stringstream ssline(line);
            if (ssline >> label1 >> numIn1 >> numOccl1 >> numRand1 >> label2) {
                // Finds the indices of label1 and label2
                auto iter1 = indicesMap.find(label1);
                if (iter1 == indicesMap.end()) i1 = -1;
                else i1 = iter1->second;
                auto iter2 = indicesMap.find(label2);
                if (iter2 == indicesMap.end()) i2 = -1;
                else i2 = iter2->second;
                std::cout << "  visited \"" << label1 << "\" [" << i1 << "] \"" << label2 << "\" [" << i2 << "]\n";
                // If both labels are found, it inserts the pair
                if (i1 >= 0 && i2 >= 0) {
                    if (i1 != i2) {
                        visitedPairs.push_back(std::make_pair(i1, i2));
                    } else {
                        // two files with the same short name are handled...
                        std::cout << "  homonymous \"" << label1 << "\": ";
                        auto range = indicesMap.equal_range(label1);
                        for (iter1 = range.first; iter1 != range.second; ++iter1) {
                            iter2 = iter1;
                            std::advance(iter2, 1);
                            for (; iter2 != range.second; ++iter2) {
                                i1 = iter1->second;
                                i2 = iter2->second;
                                if (i1 > i2) std::swap(i1, i2);
                                visitedPairs.push_back(std::make_pair(i1, i2));
                                std::cout << " (" << i1 << "," << i2 << ") ";
                            }
                        }
                        std::cout << std::endl;
                    }
                }
            }
        }
        resumeFile.close();
        outputFile << "# RESUMING" << std::endl;

        // Finds the set difference
        std::sort(inputPairs.begin(), inputPairs.end());
        std::sort(visitedPairs.begin(), visitedPairs.end());
        std::set_difference(inputPairs.begin(), inputPairs.end(),
                visitedPairs.begin(), visitedPairs.end(),
                std::back_inserter(outputPairs));

        std::cout << "Remaining pairs:\n";
        for (auto& p : outputPairs) {
            std::cout << " " << p.first << ", " << p.second << ": \"" << getShortName(inputFilenames[p.first]) << "\", \"" << getShortName(inputFilenames[p.second]) << "\"\n";
        }
        std::cout << "All pairs " << inputPairs.size() << ", visited pairs " << visitedPairs.size() << ", remaining pairs " << outputPairs.size() << std::endl;
    }

    std::string getPrefix(std::string filename) {
        // Strips filename of the path 
        expfs::path filepath(filename);
        std::string name = filepath.filename().string();
        std::string prefix;
        //  std::cout << "  name: \"" << name << "\"\n";

        // Finds the prefix
        size_t pos = name.find_first_of('_');
        if (pos != std::string::npos) {
            prefix = name.substr(0, pos);
        } else {
            prefix = name;
        }
        return prefix;
    }

    std::string getShortName(std::string filename) {
        std::stringstream ss;
        std::string prefix = getPrefix(filename);
        expfs::path filenamePath = filename;
        filename = filenamePath.filename().string();
        // Computes a digest on the string
        unsigned int h = 19;
        for (int i = 0; i < filename.length(); ++i) {
            h = ((h * 31) + (unsigned int) filename[i]) % 97;
        }
        //  std::cout << "\nglob \"" << filenamePath.string() << "\" filename \"" << filename << "\" hash " << h << std::endl;
        ss << prefix << "_" << std::setw(2) << std::setfill('0') << h;
        return ss.str();
    }

    std::string getLeafDirectory(std::string filename) {
        expfs::path filenamePath = filename;
        std::string parent = filenamePath.parent_path().string();
        size_t pos = parent.find_last_of('/');
        std::string leafDir = "";
        if (pos != std::string::npos) {
            leafDir = parent.substr(pos + 1, parent.length());
        }
        return leafDir;
    }

}