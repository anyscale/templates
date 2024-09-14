package maketmpl

import (
	"archive/zip"
	"fmt"
	"io"
	"os"
	"path/filepath"

	yaml "gopkg.in/yaml.v2"
)

type builder struct {
	baseDir string
	tmplDir string

	tmpl *Template
}

func newBuilder(t *Template, baseDir string) *builder {
	tmplDir := filepath.Join(baseDir, t.Dir)

	return &builder{tmpl: t, baseDir: baseDir, tmplDir: tmplDir}
}

func checkIsDir(path string) error {
	fi, err := os.Stat(path)
	if err != nil {
		return err
	}
	if !fi.IsDir() {
		return fmt.Errorf("%s is not a directory", path)
	}
	return nil
}

func addFileToZip(z *zip.Writer, file, path string) error {
	f, err := os.Open(file)
	if err != nil {
		return fmt.Errorf("open file %q: %w", file, err)
	}
	defer f.Close()

	w, err := z.Create(path)
	if err != nil {
		return fmt.Errorf("create file in zip: %w", err)
	}
	if _, err := io.Copy(w, f); err != nil {
		return fmt.Errorf("copy file to zip: %w", err)
	}
	return nil
}

func (b *builder) build(outputDir string) error {
	if err := checkIsDir(b.tmplDir); err != nil {
		return fmt.Errorf("check template input dir: %w", err)
	}

	// The name of the release zip file.
	const releaseDotZip = "release.zip"

	// A generated markdown file for GitHub rendering in the input directory,
	// will not be included in the release zip.
	const readmeDotMD = "README.md"
	const readmeNotebook = "README.ipynb"

	var files []string
	if err := filepath.WalkDir(
		b.tmplDir, func(p string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			relPath, err := filepath.Rel(b.tmplDir, p)
			if err != nil {
				return fmt.Errorf("get relative path for %q: %w", p, err)
			}
			if relPath == readmeDotMD {
				return nil
			}
			files = append(files, relPath)
			return nil
		},
	); err != nil {
		return fmt.Errorf("walk template dir: %w", err)
	}

	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	zipFile, err := os.Create(filepath.Join(outputDir, releaseDotZip))
	if err != nil {
		return fmt.Errorf("create release zip file: %w", err)
	}
	defer zipFile.Close()

	z := zip.NewWriter(zipFile)
	for _, f := range files {
		if err := addFileToZip(z, filepath.Join(b.tmplDir, f), f); err != nil {
			return fmt.Errorf("add file to zip: %w", err)
		}
	}
	if err := z.Close(); err != nil {
		return fmt.Errorf("close zip writer: %w", err)
	}
	if err := zipFile.Sync(); err != nil {
		return fmt.Errorf("flush zip file to storage: %w", err)
	}
	return nil
}

func readTemplates(yamlFile string) ([]*Template, error) {
	var tmpls []*Template

	bs, err := os.ReadFile(yamlFile)
	if err != nil {
		return nil, fmt.Errorf("read file %q: %w", yamlFile, err)
	}
	if err := yaml.Unmarshal(bs, &tmpls); err != nil {
		return nil, fmt.Errorf("unmarshal yaml: %w", err)
	}
	return tmpls, nil
}

// BuildAll builds all the templates defined in the YAML file.
func BuildAll(yamlFile, baseDir, outputDir string) error {
	tmpls, err := readTemplates(yamlFile)
	if err != nil {
		return fmt.Errorf("read templates: %w", err)
	}

	for _, t := range tmpls {
		b := newBuilder(t, baseDir)
		tmplOutputDir := filepath.Join(outputDir, t.Name)
		if err := b.build(tmplOutputDir); err != nil {
			return fmt.Errorf("build template %q: %w", t.Name, err)
		}
	}

	return nil
}
