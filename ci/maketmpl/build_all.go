package maketmpl

import (
	"fmt"
	"log"
	"path/filepath"
)

// BuildAll builds all the templates defined in the YAML file.
func BuildAll(yamlFile, baseDir, outputDir string) error {
	tmpls, err := readTemplates(yamlFile)
	if err != nil {
		return fmt.Errorf("read templates: %w", err)
	}

	for _, t := range tmpls {
		log.Println("Building template:", t.Name)
		b := newBuilder(t, baseDir)
		tmplOutputDir := filepath.Join(outputDir, t.Name)
		if err := b.build(tmplOutputDir); err != nil {
			return fmt.Errorf("build template %q: %w", t.Name, err)
		}
	}

	return nil
}
