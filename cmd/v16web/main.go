package main

import (
	"log"

	"github.com/MetaLoan/ponyv2/internal/v16web"
)

func main() {
	app, err := v16web.NewApp()
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("v16 web listening on %s", app.Config.Addr)
	if err := app.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}
