CC = clang++
CFLAGS = -O3 -std=c++11 -lOpenCL -fsycl
TARGET = d2q9-bgk

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cpp

clean:
	$(RM) $(TARGET) av_vels.dat final_state.dat
