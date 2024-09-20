package maketmpl

import (
	"fmt"
	"os"
	"path/filepath"
)

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

type builder struct {
	baseDir string
	tmplDir string

	tmpl *Template
}

func newBuilder(t *Template, baseDir string) *builder {
	tmplDir := filepath.Join(baseDir, t.Dir)
	return &builder{tmpl: t, baseDir: baseDir, tmplDir: tmplDir}
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
		res, err := readmeFromNotebook(nb)
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
	if err := buildZip(b.tmplDir, files, zipFile); err != nil {
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
