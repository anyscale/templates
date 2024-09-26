package maketmpl

import (
	"archive/zip"
	"bytes"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestCheckIsDir(t *testing.T) {
	tmp := t.TempDir()

	if err := checkIsDir(tmp); err != nil {
		t.Errorf("checkIsDir(%q): %v", tmp, err)
	}

	f := filepath.Join(tmp, "file")
	if err := os.WriteFile(f, nil, 0o644); err != nil {
		t.Fatalf("os.WriteFile(%q): %v", f, err)
	}

	if err := checkIsDir(f); err == nil {
		t.Errorf("checkIsDir(%q): want error, got nil", f)
	}

	if err := checkIsDir(filepath.Join(tmp, "not-exist")); err == nil {
		t.Errorf("checkIsDir(%q): want error, got nil", f)
	}
}

func TestAddFileToZip(t *testing.T) {
	tmp := t.TempDir()

	name := "README.md"
	md := filepath.Join(tmp, name)
	content := []byte("hello, world")
	if err := os.WriteFile(md, content, 0600); err != nil {
		t.Fatalf("write file: %v", err)
	}

	zipFile := filepath.Join(tmp, "a.zip")
	f, err := os.Create(zipFile)
	if err != nil {
		t.Fatalf("create zip file: %v", err)
	}

	w := zip.NewWriter(f)
	if err := addFileToZip(w, md, name); err != nil {
		t.Fatalf("addFileToZip: %v", err)
	}

	if err := w.Close(); err != nil {
		t.Fatalf("close zip writer: %v", err)
	}

	r, err := zip.OpenReader(zipFile)
	if err != nil {
		t.Fatalf("open zip file: %v", err)
	}
	defer r.Close()

	if len(r.File) != 1 {
		t.Fatalf("want output zip file to have 1 file, got %d", len(r.File))
	}

	rc, err := r.File[0].Open()
	if err != nil {
		t.Fatalf("open file in zip: %v", err)
	}

	got, err := io.ReadAll(rc)
	if err != nil {
		t.Fatalf("read file in zip: %v", err)
	}

	if err := rc.Close(); err != nil {
		t.Fatalf("close file in zip: %v", err)
	}

	if !bytes.Equal(got, content) {
		t.Errorf("content in zip file: want %q, got %q", content, got)
	}
}

func TestBuildZip(t *testing.T) {
	tmp := t.TempDir()

	tmplDir := filepath.Join(tmp, "tmpl")
	if err := os.Mkdir(tmplDir, 0700); err != nil {
		t.Fatalf("create directory %q: %v", tmplDir, err)
	}

	fileNames := []string{"a.txt", "b.txt", "sub/c.txt"}
	var files []*zipFile
	for _, f := range fileNames {
		content := []byte(f)
		dir := filepath.Dir(f)
		if dir != "." {
			if err := os.MkdirAll(
				filepath.Join(tmplDir, dir), 0700,
			); err != nil {
				t.Fatalf("create directory for %q: %v", f, err)
			}
		}

		if err := os.WriteFile(
			filepath.Join(tmplDir, f), content, 0600,
		); err != nil {
			t.Fatalf("create file %q: %v", f, err)
		}

		files = append(files, &zipFile{path: f})
	}

	z := filepath.Join(tmp, "out.zip")
	if err := buildZip(tmplDir, files, z); err != nil {
		t.Fatalf("buildZip: %v", err)
	}

	got := make(map[string][]byte)

	r, err := zip.OpenReader(z)
	if err != nil {
		t.Fatalf("open zip file: %v", err)
	}
	defer r.Close()

	for _, f := range r.File {
		rc, err := f.Open()
		if err != nil {
			t.Fatalf("open file in zip: %v", err)
		}

		content, err := io.ReadAll(rc)
		if err != nil {
			t.Fatalf("read file in zip: %v", err)
		}

		if err := rc.Close(); err != nil {
			t.Fatalf("close file in zip: %v", err)
		}

		got[f.Name] = content
	}

	if len(got) != len(fileNames) {
		t.Fatalf("want %d files in zip, got %d", len(fileNames), len(got))
	}

	for _, f := range fileNames {
		content, ok := got[f]
		if !ok {
			t.Errorf("want file %q in zip, not found", f)
		}
		if !bytes.Equal(content, []byte(f)) {
			t.Errorf("content of %q: want %q, got %q", f, f, content)
		}
	}
}
