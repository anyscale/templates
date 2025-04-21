package maketmpl

import (
	"errors"
	"fmt"
	"log"
	"path/filepath"
)

// BuildAll builds all the templates defined in the YAML file.
func BuildAll(yamlFile, baseDir, outputDir string) error {
	return buildWithFilter(yamlFile, baseDir, outputDir, nil)
}

// Build builds a single template.
func Build(yamlFile, tmplName, baseDir, outputDir string) error {
	return buildWithFilter(yamlFile, baseDir, outputDir, func(tmpl *Template) bool {
		return tmpl.Name == tmplName
	})
}

var errNoTemplateBuilt = errors.New("no template built")

func buildWithFilter(
	yamlFile, baseDir, outputDir string,
	filter func(tmpl *Template) bool,
) error {
	tmpls, err := readTemplates(yamlFile)
	if err != nil {
		return fmt.Errorf("read templates: %w", err)
	}

	buildCount := 0

	for _, t := range tmpls {
		if filter != nil && !filter(t) {
			continue
		}

		log.Println("Building template:", t.Name)
		b := newBuilder(t, baseDir)
		tmplOutputDir := filepath.Join(outputDir, t.Name)
		if err := b.build(tmplOutputDir); err != nil {
			return fmt.Errorf("build template %q: %w", t.Name, err)
		}

		buildCount++
	}

	if buildCount == 0 {
		return errNoTemplateBuilt
	}

	return nil
}
