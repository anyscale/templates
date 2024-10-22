package maketmpl

import (
	"os"
	"path/filepath"
	"testing"
)

func TestInlineImgSrc(t *testing.T) {
	imageFile := []byte("fakeimg")

	tests := []struct {
		file           string
		want           string
		skipFileCreate bool
		wantErr        bool
	}{
		{file: "img.png", want: "data:image/png;base64,ZmFrZWltZw=="},
		{file: "img.jpeg", want: "data:image/jpeg;base64,ZmFrZWltZw=="},
		{file: "img.jpg", want: "data:image/jpeg;base64,ZmFrZWltZw=="},
		{file: "img.gif", want: "data:image/gif;base64,ZmFrZWltZw=="},
		{
			file:           "http://example.com/a.png",
			want:           "http://example.com/a.png",
			skipFileCreate: true,
		},
		{
			file:           "https://example.com/a.png",
			want:           "https://example.com/a.png",
			skipFileCreate: true,
		},
		{file: "img.svg", wantErr: true},
		{file: "not-exist.png", skipFileCreate: true, wantErr: true},
	}

	for _, test := range tests {
		t.Run(test.file, func(t *testing.T) {
			tmp := t.TempDir()

			if !test.skipFileCreate {
				img := filepath.Join(tmp, test.file)
				if err := os.WriteFile(img, imageFile, 0o644); err != nil {
					t.Fatalf("write fake png: %v", err)
				}
			}

			got, err := inlineImgSrc(tmp, test.file)
			if test.wantErr {
				if err == nil {
					t.Errorf("inlineImgSrc(%q): want error, got nil", test.file)
				}
				return
			}

			if err != nil {
				if !test.wantErr {
					t.Fatalf("inlineImgSrc(%q): %v", test.file, err)
				}
			} else if got != test.want {
				t.Errorf(
					"inlineImgSrc(%q) = %q, want %q",
					test.file, got, test.want,
				)
			}
		})
	}
}
