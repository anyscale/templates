package main

import (
	"flag"
	"log"

	"github.com/anyscale/templates/ci/maketmpl"
)

func main() {
	base := flag.String("base", ".", "base directory")
	output := flag.String("output", "_build", "output directory")
	buildFile := flag.String("build", "BUILD.yaml", "build file")

	flag.Parse()

	if err := maketmpl.BuildAll(*buildFile, *base, *output); err != nil {
		log.Fatal(err)
	}
}
