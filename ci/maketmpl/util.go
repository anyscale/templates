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

func addFileToZip(z *zip.Writer, file, pathInZip string) error {
	f, err := os.Open(file)
	if err != nil {
		return fmt.Errorf("open file %q: %w", file, err)
	}
	defer f.Close()

	w, err := z.Create(pathInZip)
	if err != nil {
		return fmt.Errorf("create file in zip: %w", err)
	}
	if _, err := io.Copy(w, f); err != nil {
		return fmt.Errorf("copy file to zip: %w", err)
	}
	return nil
}

func buildZip(dir string, files []string, out string) error {
	outFile, err := os.Create(out)
	if err != nil {
		return fmt.Errorf("create release zip file: %w", err)
	}
	defer outFile.Close()

	z := zip.NewWriter(outFile)
	for _, f := range files {
		if err := addFileToZip(z, filepath.Join(dir, f), f); err != nil {
			return fmt.Errorf("add file to zip: %w", err)
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
