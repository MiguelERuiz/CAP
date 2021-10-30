# ==============================================================================
NAME=black-scholes
PRJ = "./Ejemplo6-Report"

TEST_CONSTANT := 90000000

SRC:=                      \
       black-scholes.c     \
       black-scholes_mkl.c \
       data_gen.c          \
       main.c

# ==============================================================================
# ############## Configure CFLAGS  #############################################
# ==============================================================================
FQOPT=f
EQCOLON= =

CC          := icc
CFLAGS      += -O2 -g -qopt-report=5
# CCFLAGS     += -qopt-report-phase$(EQCOLON)vec
CFLAGS      += -I./

PREC ?= s
ifeq ($(PREC),d)
else
    CFLAGS += -D__DO_FLOAT__
endif

TARGET_ARCH ?= AVX

ifeq ($(TARGET_ARCH),SSE)
    CFLAGS += -$(QOPT)xSSE4.2
endif
ifeq ($(TARGET_ARCH),AVX)
    CFLAGS += -$(QOPT)xAVX
endif
ifeq ($(TARGET_ARCH),AVX2)
    CFLAGS += -$(QOPT)xCORE-AVX2
endif
ifeq ($(TARGET_ARCH), HOST)
    CFLAGS += -xHost
endif


ifeq ($(OS),Win)
    FQOPT=Q
endif

ACC ?= ep
ifeq ($(ACC),ha)
    CFLAGS += -$(FQOPT)imf-precision$(EQCOLON)high -D_VML_ACCURACY_HA_
endif
ifeq ($(ACC),la)
    CFLAGS += -$(FQOPT)imf-precision$(EQCOLON)medium -D_VML_ACCURACY_LA_
endif
ifeq ($(ACC),ep)
    CFLAGS += -$(FQOPT)imf-precision$(EQCOLON)low -$(FQOPT)imf-domain-exclusion$(EQCOLON)31 -D_VML_ACCURACY_EP_
endif

# ==============================================================================
# ############## Define make rules #############################################
# ==============================================================================

# Enable -qmkl instead of -mkl to avoid warnings
$(NAME): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -qmkl -o $(NAME)


run: $(NAME)
	./black-scholes $(TEST_CONSTANT)

clean:
	rm -rf $(NAME) *.o *.out *.optrpt $(PRJ)


survey:
	advixe-cl -collect survey -project-dir $(PRJ) -- ./$(NAME) $(TEST_CONSTANT)

roofline:
	advixe-cl -collect survey -project-dir $(PRJ) -- ./$(NAME) $(TEST_CONSTANT)
	advixe-cl -collect tripcounts -flop -project-dir $(PRJ) -- ./$(NAME) $(TEST_CONSTANT)

open-gui:
	advixe-gui $(PRJ)/$(PRJ).advixeproj >/dev/null 2>&1 &
