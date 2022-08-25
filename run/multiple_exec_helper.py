import os
import sys

#nohup!


#folder_to_crawl_path = "/home/ernest/Downloads/mpeg7_dataset/mpeg7_point_tests/"
folder_to_crawl_path = "/home/rimlab/Downloads/mpeg7_point_tests/"
cfg_filename = "../cfg/gpu_fullArsIsoMpeg7.cfg"



folders = os.listdir(folder_to_crawl_path)

folder_counter = 0


for f in folders:
	# if folder_counter<8:	
	print(f)
	command = "nohup ./gpu_fullArsIsoMpeg7 -cfg " + cfg_filename + " -in " + folder_to_crawl_path + f + "/"
	output_redir =  " > out_mepg7firsttest_" + str(folder_counter) + ".log"
	print (command+output_redir)
	os.system(command+output_redir)
	folder_counter = folder_counter + 1	
