CARGO := cargo --offline

#TARGET := x86_64-unknown-linux-gnu
#CARGO_TARGET := --target $(TARGET)
CARGO_TARGET :=

.PHONY: all dev debug build rel release clean distclean

all: debug

dev: debug

debug:
	$(CARGO) build $(CARGO_TARGET) --lib --bins --examples

build: release

rel: release

release:
	$(CARGO) build $(CARGO_TARGET) --release --lib --bins --examples

clean: distclean

distclean:
	rm -r target
