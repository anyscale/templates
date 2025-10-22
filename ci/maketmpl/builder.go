package maketmpl

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
)

// Input files.
const (
	// The main notebook file for the template.
	readmeDotNotebook = "README.ipynb"

	// The main README markdown file.
	readmeDotMD = "README.md"
)

// Output files.
const (
	// The name of the build result zip file.
	buildDotZip = "build.zip"

	// The name of the preview result zip file.
	previewDotZip = "preview.zip"

	// The name of the tempalte metadata JSON file.
	rayAppDotJSON = "ray-app.json"

	// A generated single page, self-contaiend markdown file for
	// documentation purposes.
	readmeDocMD = "README.doc.md"

	// A generated markdown file for GitHub rendering in the output directory.
	readmeGitHubMD = "README.github.md"

	// A markdown file converted from the notebook.
	// This might be copied back in the GitHub directory.
	readmeNotebookGitHubMD = "README.nb.github.md"
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
			files = append(files, relPath)
			return nil
		},
	); err != nil {
		return nil, err
	}

	return files, nil
}

func (b *builder) build(outputDir string) error {
	if err := checkIsDir(b.tmplDir); err != nil {
		return fmt.Errorf("check template input dir: %w", err)
	}

	// List all files in the template directory.
	files, err := b.listFiles()
	if err != nil {
		return fmt.Errorf("list files: %w", err)
	}
	fileSet := make(map[string]struct{})
	for _, f := range files {
		fileSet[f] = struct{}{}
	}

	// Check if README markdown and/or notebook files are present.
	// If yes, load them in.
	var readmeMD, readmeNB *readmeFile
	if _, found := fileSet[readmeDotNotebook]; found {
		nb := filepath.Join(b.tmplDir, readmeDotNotebook)
		res, err := readmeFromNotebook(nb)
		if err != nil {
			return fmt.Errorf("build readme: %w", err)
		}
		readmeNB = res
	}
	if _, found := fileSet[readmeDotMD]; found {
		res, err := readReadmeFile(filepath.Join(b.tmplDir, readmeDotMD))
		if err != nil {
			return fmt.Errorf("read readme file: %w", err)
		}
		readmeMD = res
	}

	// Build the meta data of the template.
	meta := &templateMeta{
		Name:                b.tmpl.Name,
		ComputeConfigBase64: make(map[string]string),
	}
	for cld, f := range b.tmpl.ComputeConfig {
		bs, err := os.ReadFile(filepath.Join(b.baseDir, f))
		if err != nil {
			return fmt.Errorf("read compute config %q: %w", f, err)
		}
		meta.ComputeConfigBase64[cld] = base64.StdEncoding.EncodeToString(bs)
	}
	if b.tmpl.ClusterEnv != nil {
		bs, err := json.Marshal(b.tmpl.ClusterEnv)
		if err != nil {
			return fmt.Errorf("marshal cluster env: %w", err)
		}
		meta.ClusterEnvBase64 = base64.StdEncoding.EncodeToString(bs)
	}

	metaEncoded, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal meta: %w", err)
	}

	// Gather all source files.
	var srcFiles []*zipFile
	srcFiles = append(srcFiles, &zipFile{
		path: path.Join(".meta", rayAppDotJSON),
		rc:   io.NopCloser(bytes.NewReader(metaEncoded)),
	})
	for _, f := range files {
		srcFiles = append(srcFiles, &zipFile{path: f})
	}

	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	// Build the release zip file.
	zipOutput := filepath.Join(outputDir, buildDotZip)
	if err := buildZip(b.tmplDir, srcFiles, zipOutput); err != nil {
		return fmt.Errorf("save release zip file: %w", err)
	}

	// Preview zip file contains all the source files except the meta file.
	// This is for the file viewer to show preview of the template content.
	var previewFiles []*zipFile
	for _, f := range files {
		previewFiles = append(previewFiles, &zipFile{path: f})
	}
	previewZipOutput := filepath.Join(outputDir, previewDotZip)
	if err := buildZip(b.tmplDir, previewFiles, previewZipOutput); err != nil {
		return fmt.Errorf("save preview zip file: %w", err)
	}

	// Write out the ray-app.json file as an independent file too.
	metaFile := filepath.Join(outputDir, rayAppDotJSON)
	if err := os.WriteFile(metaFile, metaEncoded, 0600); err != nil {
		return fmt.Errorf("write meta file: %w", err)
	}

	// Write out README files of various forms...
	var readme *readmeFile
	if readmeMD != nil {
		// if markdown README file presents, use the markdown version.
		readme = readmeMD
	} else if readmeNB != nil {
		// Otherwise, if the notebook README file presents, use the
		// notebook version.
		readme = readmeNB
	}

	if readme != nil {
		if err := readme.writeReleaseMD(
			filepath.Join(outputDir, readmeDocMD), b.tmplDir,
		); err != nil {
			return fmt.Errorf("write release readme file: %w", err)
		}
	}

	if readmeMD != nil {
		// This is a cleaned up version of the README file for GitHub
		// rendering. Likely won't be used, but we generate it just for
		// reference.
		if err := readme.writeGitHubMD(
			filepath.Join(outputDir, readmeGitHubMD),
		); err != nil {
			return fmt.Errorf("write github readme file: %w", err)
		}
	}

	// Write out README converted from notebook.
	if readmeNB != nil {
		if err := readmeNB.writeGitHubMD(
			filepath.Join(outputDir, readmeNotebookGitHubMD),
		); err != nil {
			return fmt.Errorf("write github readme file: %w", err)
		}
	}

	return nil
}
