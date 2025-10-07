module github.com/lee101/gobed

go 1.23.0

toolchain go1.25.0

require (
	github.com/fatih/color v1.18.0
	github.com/lee101/gobed/cuvs_cagra v0.0.0-00010101000000-000000000000
	github.com/sugarme/gotch v0.9.0
	github.com/sugarme/tokenizer v0.3.0
	golang.org/x/sys v0.25.0
)

require (
	github.com/mattn/go-colorable v0.1.13 // indirect
	github.com/mattn/go-isatty v0.0.20 // indirect
)

require (
	github.com/daulet/tokenizers v1.23.0
	github.com/emirpasic/gods v1.18.1 // indirect
	github.com/mitchellh/colorstring v0.0.0-20190213212951-d06e56a500db // indirect
	github.com/patrickmn/go-cache v2.1.0+incompatible // indirect
	github.com/rivo/uniseg v0.4.7 // indirect
	github.com/schollz/progressbar/v2 v2.15.0 // indirect
	github.com/sugarme/regexpset v0.0.0-20200920021344-4d4ec8eaf93c // indirect
	golang.org/x/text v0.25.0 // indirect
)

exclude github.com/sugarme/gotch v0.9.1

replace github.com/lee101/gobed/cuvs_cagra => ./cuvs_cagra
