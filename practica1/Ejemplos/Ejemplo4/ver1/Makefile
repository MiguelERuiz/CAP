CXX = icpc
COMPFLAGS = -g -std=c++11 -O2 -I. -xHost
INCLUDES = 

CXXFLAGS = $(COMPFLAGS) 

SOURCES = main.cpp

ADVPRJ = "./adv-ver1"

ifeq ($(REPORT), yes)
	CXXFLAGS+=-qopt-report=5
ifeq ($(FILTER), yes) 
	CXXFLAGS+=-qopt-report-phase=vec -qopt-report-filter="main.cc,15-25"
endif
endif

.SUFFIXES: .o .cpp

##########################################
OBJSC = $(SOURCES:.cpp=.o)
##########################################

EXEC=coulomb

all: clean cpu

%.o: %.cpp
	$(info )
	$(info Compiling the object file for CPU: )
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ 

cpu: $(OBJSC)
	$(info )
	$(info Linking the CPU executable:)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXEC) $(OBJSC)
	
run: 
	$(info )
	$(info Run the default test case on CPU: )
	./$(EXEC)
	
clean: 
	rm -f $(OBJSC) $(EXEC) *.optrpt 

#----------------------------------------------------------------

survey:
	advixe-cl -collect survey -project-dir $(ADVPRJ) -- ./$(EXEC)

roofline:
	advixe-cl -collect survey -project-dir $(ADVPRJ) -- ./$(EXEC) 
	advixe-cl -collect tripcounts -flop -project-dir $(ADVPRJ) -- ./$(EXEC) 

map:
	advixe-cl -collect map -mark-up-list=1 -project-dir $(ADVPRJ) -- ./$(EXEC)  

open-gui:
	advixe-gui $(ADVPRJ)/$(ADVPRJ).advixeproj >/dev/null 2>&1 &

vtune-aps:
	aps ./$(EXEC)

vtune-hotspots: 
	vtune -collect hotspots -r $(ADVPRJ) -- ./$(EXEC)

vtune-uarch:
	vtune -collect uarch-exploration -r $(ADVPRJ) -- ./$(EXEC)


vtune-memory-access:
	vtune -collect memory-access  -knob analyze-mem-objects=true -knob dram-bandwidth-limits=true -r $(ADVPRJ) -- ./$(EXEC)

vtune-open-gui:
	vtune-gui $(ADVPRJ)/$(ADVPRJ).vtune >/dev/null 2>&1 &

clean-results:
	rm -rf $(ADVPRJ) *aps*




