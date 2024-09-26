package maketmpl

import (
	"archive/zip"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

func checkIsDir(path string) error {
	stat, err := os.Stat(path)
	if err != nil {
		return err
	}
	if !stat.IsDir() {
		return fmt.Errorf("%s is not a directory", path)
	}
	return nil
}

func addToZip(z *zip.Writer, r io.Reader, pathInZip string) error {
	w, err := z.Create(pathInZip)
	if err != nil {
		return fmt.Errorf("create file in zip: %w", err)
	}
	if _, err := io.Copy(w, r); err != nil {
		return err
	}
	return nil
}

func addFileToZip(z *zip.Writer, file, pathInZip string) error {
	f, err := os.Open(file)
	if err != nil {
		return fmt.Errorf("open file %q: %w", file, err)
	}
	defer f.Close()

	return addToZip(z, f, pathInZip)
}

type zipFile struct {
	// Path to use in the zip file.
	path string

	// Optional. If set, the content will be read from this reader.
	rc io.ReadCloser
}

func buildZip(srcDir string, files []*zipFile, out string) error {
	outFile, err := os.Create(out)
	if err != nil {
		return fmt.Errorf("create release zip file: %w", err)
	}
	defer outFile.Close()

	z := zip.NewWriter(outFile)
	for _, f := range files {
		if f.rc == nil {
			src := filepath.Join(srcDir, f.path)
			if err := addFileToZip(z, src, f.path); err != nil {
				return fmt.Errorf("add file %q to zip: %w", f, err)
			}
		} else {
			if err := addToZip(z, f.rc, f.path); err != nil {
				return fmt.Errorf("add %q to zip: %w", f.path, err)
			}
		}
	}
	if err := z.Close(); err != nil {
		return fmt.Errorf("close zip writer: %w", err)
	}
	if err := outFile.Sync(); err != nil {
		return fmt.Errorf("flush zip file to storage: %w", err)
	}
	return nil
}
