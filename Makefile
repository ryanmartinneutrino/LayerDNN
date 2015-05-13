#Generic Makefile .
#Each doublet of files (X.cc, X.hh) is compiled together into a object file
#and all object files are linked into a library (.so) in the current directory.
#A class should be defined by the 2 files (.hh, .cc). If you have 2 classes, then there
#there will be 4 files...  
#All .cc files will be compiled to .o and linked into a .so file, regardless
#of whether they define classes.
#Any .cc file that is found in a subdirectory called main/ will be assumed
#to contain a main() function. Each of these is linked to the .so file to 
#create an executable of the same name.
#After a make clean, make has to be run twice to make the library.
#R. Martin

SrcSuf = cpp #hardcoded in some spots... 
ObjSuf = o
ExeSuf = 
CXX = g++
LD = g++

CXXFLAGS = -fPIC -Wall -O3 -std=c++11
LDFLAGS  = -fPIC
SOFLAGS  = -shared

ALLINC= -Isrc/
ALLLIB= -lfftw3 -lm -lgmpxx -lgmp

EXESRC := $(wildcard main/*.$(SrcSuf))
EXEOBJ := $(EXESRC:.cpp=.o)
EXE :=  $(patsubst main/%.$(ObjSuf), %$(ExeSuf), $(wildcard  main/*.$(ObjSuf)))

SRC := $(wildcard src/*.cpp)
OBJ := $(SRC:.cpp=.o)

#This is the name of the library that will be built:
LIBRARY = liblayerdnn.so

all : $(OBJ) $(EXEOBJ) dynamiclib $(EXE)  


%.$(ObjSuf): %.$(SrcSuf)
	$(CXX) -c $(ALLINC) $(CXXFLAGS) $< -o $@ 

%$(ExeSuf): main/%.$(ObjSuf)
	@echo linking executable
	$(LD) $(LDFLAGS) $< -o $@ $(ALLLIB) $(HOME)/usr/lib/$(LIBRARY) 
	cp $@ $(HOME)/usr/bin/.

dynamiclib: $(OBJ) $(HEADERFILES) 
	@echo "Generating library $(LIBRARY)..."
	$(LD) $(SOFLAGS) $(LDFLAGS) $(OBJ) -o $(LIBRARY)
	#cp $(LIBRARY) $(HOME)/usr/lib/.

.PHONY : clean

clean:
	rm $(EXEOBJ) $(OBJ) $(EXE) $(LIBRARY)
