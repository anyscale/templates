package maketmpl

import (
	"archive/zip"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"

	yaml "gopkg.in/yaml.v2"
)

type builder struct {
	baseDir string
	tmplDir string

	tmpl *Template
}

const (
	// The name of the release zip file.
	releaseDotZip = "release.zip"

	// The main notebook file for the template.
	readmeNotebook = "README.ipynb"

	// A generated markdown file for GitHub rendering in the input directory,
	// will not be included in the release zip.
	readmeDotMD = "README.md"

	// A generated markdown file for GitHub rendering in the output directory.
	readmeGitHubMD = "README.github.md"
)

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

func (b *builder) listFiles() ([]string, error) {
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
		return nil, err
	}

	return files, nil
}

func (b *builder) buildReleaseZip(path string, files []string) error {
	zipFile, err := os.Create(path)
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

func hasReadmeNotebook(files []string) bool {
	for _, f := range files {
		if f == readmeNotebook {
			return true
		}
	}
	return false
}

func (b *builder) build(outputDir string) error {
	if err := checkIsDir(b.tmplDir); err != nil {
		return fmt.Errorf("check template input dir: %w", err)
	}

	files, err := b.listFiles()
	if err != nil {
		return fmt.Errorf("list files: %w", err)
	}

	var readme *readmeFile
	if hasReadmeNotebook(files) {
		// Generate a markdown file for GitHub rendering.
		nb := filepath.Join(b.tmplDir, readmeNotebook)
		res, err := buildReadme(nb)
		if err != nil {
			return fmt.Errorf("build readme: %w", err)
		}
		readme = res
	}

	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	// Build the release zip file.
	zipFile := filepath.Join(outputDir, releaseDotZip)
	if err := b.buildReleaseZip(zipFile, files); err != nil {
		return fmt.Errorf("save release zip file: %w", err)
	}

	if readme != nil {
		if err := readme.writeGitHubMD(
			filepath.Join(outputDir, readmeGitHubMD), b.tmplDir,
		); err != nil {
			return fmt.Errorf("write github readme file: %w", err)
		}

		if err := readme.writeReleaseMD(
			filepath.Join(outputDir, readmeDotMD), b.tmplDir,
		); err != nil {
			return fmt.Errorf("write release readme file: %w", err)
		}
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
		log.Println("Building template:", t.Name)
		b := newBuilder(t, baseDir)
		tmplOutputDir := filepath.Join(outputDir, t.Name)
		if err := b.build(tmplOutputDir); err != nil {
			return fmt.Errorf("build template %q: %w", t.Name, err)
		}
	}

	return nil
}
