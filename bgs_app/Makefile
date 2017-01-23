OBJ = vibe.o utils.o bgs_app.o
INCLUDE = /usr/local/include
OPENCVLIB = -L/usr/local/lib -L/usr/local/cuda-7.0/lib -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab -lcufft -lnpps -lnppi -lnppc -lcudart -lrt -lpthread -lm -ldl
CPP_FLAG = -std=c++11
Compile = g++
bgs_app : $(OBJ)
	g++ -o bgs_app $(OBJ) -I $(INCLUDE) $(OPENCVLIB) $(CPPFLAG)
bgs_app.o : bgs_app.cpp utils.h vibe.h
	g++ -c bgs_app.cpp $(CPP_FLAG)
vibe.o : vibe.cpp vibe.h utils.h
	g++ -c vibe.cpp $(CPP_FLAG)
utils.o  : utils.cpp utils.h
	g++ -c utils.cpp $(CPP_FLAG)
clean : 
	rm bgs_app $(OBJ)
