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

	args := flag.Args()
	if len(args) == 0 {
		if err := maketmpl.BuildAll(*buildFile, *base, *output); err != nil {
			log.Fatal(err)
		}
	} else if len(args) == 1 {
		if err := maketmpl.Build(*buildFile, args[0], *base, *output); err != nil {
			log.Fatal(err)
		}
	} else {
		log.Fatal("too many arguments")
	}
}
