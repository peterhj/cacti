TARGET := x86_64-unknown-linux-gnu

CARGO := cargo --offline
CARGO_TARGET := --target $(TARGET)

.PHONY: all clean

all:
	$(CARGO) build $(CARGO_TARGET) --release

clean:
	rm -r target
