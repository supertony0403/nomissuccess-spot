# Brief: nomissuccess-Spot „Nachtschicht“ (Oktober 2026)

Status: **Entwurf, wartet auf Anthonys OK.** Gebaut wird erst nach „go“.
Angepasst aus dem Spec-Ad-Prompt (Riccardo-Bosso-Vorlage) für Linux, DaVinci Resolve und Gemini.

---

## 1. Auftrag (angepasster Prompt)

| Feld | Wert |
|---|---|
| Marke | **nomissuccess**, IT-Dienstleister aus Schweinfurt, Inhaber Simon Socha |
| Referenz | keine Vorlage, Richtung Spotify-, Shopify- und Huel-Spots, Apple-Keynote-Produktfilme |
| Idee | drei Ideen unten, Empfehlung: **A „Nachtschicht“** |
| Skript | schreibe ich, nur aus belegten Website-Aussagen und Anthonys Vorgaben |
| Stimme | **Google Gemini TTS** (eigener API-Schlüssel), vier Stimmproben zur Auswahl |
| Länge | **≥ 2:00**, Ziel 2:20–2:40, plus ca. 2 s Abspann |
| Werkzeug | **DaVinci Resolve Studio 21.1.1** als Zentrale (Edit, Fusion, Color, Fairlight, Deliver), **Blender 5.2** für die 3D-Heldenshots |
| Formate | 1920×1080 und 1080×1920, **60 fps**, je mit Ton und stumm |

### Nicht machen
- **Nicht der Look der bisherigen Spots** (auch nicht vom ersten nomissuccess-Spot). Der war „zu rentner“. Also keine Schrift auf dunkler Fläche mit Überblendung, keine statisch zentrierten Titel und keine Wort-Slams, Zoom-Punches, RGB-Splits, Scanlines oder Blitze von dort.
- **Kein NIS2.** Das bewerben wir nicht.
- **Keine Herstellerlogos** (Simons Rechtsregel vom 15.07.). FortiGate, OPNsense, Proxmox usw. nur als Textnennung.
- Nichts erfinden: keine Kunden, keine Referenzen, keine Prozent- oder Euro-Zahlen, die nicht auf der Website stehen.
- Verbotene Formulierungen aus dem Unternehmensprofil: *Rechenzentrum, Proxmox-Cluster, 24/7, rund um die Uhr, NIS2, Partner von, zertifizierter Partner, IPSec-Kopplung, Standortkopplung, DDoS, Geschäftsführer.*
- Pexels-Fotos sind **Symbolbilder.** Nie mit „unser eigenes …“ oder „unser Techniker“ überschreiben. „Kein Mockup“ steht nur über **echten** Screenshots (Grafana, Proxmox, PBS).
- Kein „KI-Look“: kein Glas-Morphismus, keine Verlaufs-Schrift in Überschriften, keine schwebenden Blobs als Mittelpunkt, keine Spiegelsymmetrie und keine Fenster-Rahmen um Screenshots.

---

## 2. Drei Ideen

**A — „Nachtschicht“ (Empfehlung).** Der Spot erzählt eine Nacht, von **03:12 bis 07:00**. Während die Stadt schläft, arbeitet die IT: Website, Firewall, Backup und Server. Jede Leistung ist eine Station der Nacht. Am Morgen läuft alles, und niemand wurde geweckt. Der Schluss ist die Website-Zeile **„Infrastruktur, die nachts niemanden weckt.“** Diese Klammer trägt zwei Minuten, ohne Angst zu machen.

**B — „Kein Baukasten“.** Ein Manifest im Gegensatz-Rhythmus: Baukasten gegen programmiert, Callcenter gegen Team, Hoffnung gegen getesteten Restore. Schnell, laut, Split-Screens. Stark für Social, aber zwei Minuten davon ermüden.

**C — „Das Band“.** Abstrakter Markenfilm in einer einzigen Kamerafahrt, die von der Website-Oberfläche Schicht für Schicht nach unten taucht: Code, Netzwerk, Firewall, Backup, Server. Bildgewaltig, aber ohne Geschichte.

**Empfehlung:** A, mit dem **Logo-Band aus C** als rotem Faden: Das „N“ des Logos ist ein Band mit Farbverlauf. Als 3D-Lichtband läuft es durch jede Szene (als Glasfaser, Datenstrom und Verbindung) und faltet sich am Ende selbst zum Logo.

---

## 3. Skript (Konzept A), in ganzen, klaren Sätzen

Rechnung: ca. 340 Wörter bei 2,4–2,6 Wörtern/s, plus Atempausen, ergibt **2:20–2:40**. Das Bild wird auf die Stimme geschnitten, nicht umgekehrt.

| ID | Szene | Text |
|---|---|---|
| v01 | 1 Nacht | Drei Uhr zwölf. Die Stadt schläft. Ihr Büro ist dunkel. |
| v02 | 1 Nacht | Aber Ihre IT schläft nie: Website, Firewall, Server, Backups. |
| v03 | 1 Nacht | Die Frage ist nur: Weckt sie heute Nacht jemanden? |
| v04 | 2 Website | Fangen wir vorne an, bei Ihrer Website. Sie ist Ihr Schaufenster, auch um drei Uhr nachts. |
| v05 | 2 Website | Wir bauen sie für Sie. Komplett programmiert. Kein Baukasten, keine Plugin-Sammlung, keine Vorlage von der Stange. |
| v06 | 2 Website | Sie lädt in unter einer Sekunde, liegt auf Servern in Deutschland und wird jeden Tag gesichert. |
| v07 | 2 Website | Und Sie zahlen monatlich: Hosting, Updates und Pflege für einen festen Betrag, ohne Überraschungen. **⟵ siehe Frage 2** |
| v08 | 2 Website | Der Quellcode gehört trotzdem Ihnen. |
| v09 | 3 Netzwerk | Hinter der Website beginnt Ihr Netzwerk. Dort entscheidet sich, ob ein Angreifer draußen bleibt. |
| v10 | 3 Netzwerk | Wir richten Ihre Firewall ein, mit FortiGate oder OPNsense, herstellerneutral und aus dem täglichen Betrieb. |
| v11 | 3 Netzwerk | Und der Zugriff aus dem Homeoffice läuft ohne klassisches VPN. Mit Zero Trust ist Ihr Firmennetz im Internet schlicht unsichtbar. Es gibt keinen offenen Port, den man scannen könnte. |
| v12 | 4 Ernstfall | Und wenn doch etwas passiert? Ransomware greift zuerst die Sicherungen an. |
| v13 | 4 Ernstfall | Darum legen wir unveränderbare Kopien an und spielen sie regelmäßig zurück. Denn ein Backup, das nie getestet wurde, ist nur eine Hoffnung. |
| v14 | 5 Betrieb | Ihre VMware-Lizenz kostet plötzlich ein Vielfaches? Wir ziehen Ihre Server auf Proxmox um, planbar, mit Testlauf und Rückweg. |
| v15 | 5 Betrieb | Updates spielen wir im laufenden Betrieb ein, mit Dashboards, die man auch ohne IT-Studium lesen kann. |
| v16 | 6 Beweis | Klingt nach Prospekt? Ist es nicht. Wir betreiben selbst, was wir Ihnen empfehlen. |
| v17 | 6 Beweis | Was Sie hier sehen, ist kein Mockup. Das ist unser eigenes Monitoring, live und im Dauerbetrieb. |
| v18 | 7 Team | Und das alles kommt von einem Team, von der Website bis zur Firewall. |
| v19 | 7 Team | Kein Callcenter, kein Ticketstapel. Sie sprechen mit den Leuten, die Ihre Systeme bauen, und bekommen innerhalb eines Werktags eine Antwort. |
| v20 | 7 Team | Am Ende gehört Ihnen die komplette Dokumentation. Keine Black Box, keine Abhängigkeit. |
| v21 | 8 Morgen | Sieben Uhr. Alles läuft. Niemand wurde geweckt. |
| v22 | 8 Morgen | nomissuccess. Infrastruktur, die nachts niemanden weckt. |
| v23 | 8 Morgen | Das Erstgespräch dauert dreißig Minuten und kostet nichts. nomissuccess punkt de. |

**Belege** (alle von nomissuccess.de): Ladezeit unter 1 s, Hosting in Deutschland, tägliche Backups und Quellcode gehört dem Kunden (Leistungsseite Web). Herstellerneutral FortiGate/OPNsense, Zero Trust „im Internet schlicht unsichtbar“ und „Ein Backup, das nie zurückgespielt wurde, ist nur eine Hoffnung“ stehen auf Startseite und Referenzarchitektur. VMware „vervielfacht“, Testlauf und Rollback-Pfad stehen auf der Migrationsseite. Ebenso belegt: „Dashboards … ohne IT-Studium lesbar“, „Wir betreiben selbst, was wir empfehlen“, „Kein Mockup“ (nur über echtem Grafana), „kein Callcenter“, „Antwort innerhalb eines Werktags“, „Keine Black Box“, Erstgespräch „30 min · 0 €“ und die Schlusszeile „Infrastruktur, die nachts niemanden weckt.“
**Neu von Anthony (05.10.):** monatliche Zahlung, komplett programmiert, kein Baukasten, ein Team.

**Aussprache für die Stimme** (Umschrift nur im Stimm-Text, nie im Bild):
nomissuccess → „No-Miss-Success“ (englisch, **bitte bestätigen**) · Proxmox → „Prox-Mox“ · OPNsense → „O-P-N-Sense“ · FortiGate → „Forti-Gate“ · VMware → „Vi-Em-Wär“ · Zero Trust → „Siro Trast“ · Ransomware → „Rensomwär“. Im Stimmtest prüfe ich, welche Umschrift Gemini wirklich braucht.

---

## 4. Ablauf (Beat Sheet)

Zeiten sind Richtwerte. Die echten Zeiten kommen aus den gemessenen Sprachdauern.
**Jede Szene hat eigene Bewegungen, nichts wird wiederholt.** Durchgehend läuft eine Motion-Graphics-Ebene mit: oben links eine Mono-Uhr, die die Nacht mitzählt, unten eine Fortschrittsleiste im Signalfaden-Verlauf (Marken-Moment 1), dazu Staubpartikel, Filmkorn und Lichtblüte auf Spitzlichtern. Spätestens alle 2 s passiert etwas Neues.

| # | Szene | ca. | Farbe | Bild — Signatur-Bewegungen |
|---|---|---|---|---|
| 1 | **03:12** | 0:00–0:14 | Nacht | Schwarz. Eine Rollzähler-Uhr dreht von 03:11:57 auf **03:12:00**, jede Ziffer mit vertikaler Bewegungsunschärfe. Das Logo-Band schneidet als hauchdünne Lichtlinie durchs Bild, die Kamera folgt ihm durch Stadtlichter im Bokeh. „Die Stadt schläft.“: die Buchstaben sacken weg und dimmen ab. Bei „dunkel“ geht Buchstabe für Buchstabe das Licht aus, mit Klick-Geräusch. Die vier Wörter Website · Firewall · Server · Backups leuchten als Status-LEDs auf. Das Fragezeichen verwandelt sich in eine Weckerglocke, die einmal zuckt. |
| 2 | **Schaufenster** | 0:14–0:50 | Rosa/Violett | Die echte nomissuccess.de-Seite steht als beleuchtetes Schaufenster in 3D-Perspektive, mit Spiegelung am Boden. Bei „komplett programmiert“ **zerlegt sich die Seite in ihre Schichten** (Raster, Typo, Bilder, echter Astro-Quellcode), die mit Tiefenunschärfe in den Raum auseinanderfliegen und mit einem Treffer wieder einrasten. „Kein Baukasten“: ein Turm aus 3D-Bausteinen mit „Plugin“, „Theme“ und „Tracking“ kippt und fällt mit Physik aus dem Bild. „< 1 s“: eine Linie rast über das Bild, die Zahl stoppt hart. „Jeden Tag gesichert“: ein Kalender hakt im Zeitraffer ab. „Monatlich“: Monatskarten Jan–Dez gleiten als Karussell durch, darauf „1 fester Betrag“. „Quellcode gehört Ihnen“: ein 3D-Schlüssel dreht sich ins Schloss, Mono-Label `git clone — Ihr Repository`. |
| 3 | **Netzwerk** | 0:50–1:15 | Blau | Die Kamera taucht **durch die Website-Pixel ins Netzwerk**: Bildpunkte werden zu Netzknoten. Die Firewall ist eine Wand aus Lichtflächen. Rote Pakete prallen funkenschlagend ab, grüne gehen durch. FortiGate und OPNsense stehen nur als Mono-Text. Zero Trust: Ein Radarstrahl sucht das Feld ab, das Firmennetz löst sich dabei ins Dunkel auf, und der Zähler steht bei **„0 offene Ports“**. Danach zieht ein Laptop ein feines Band zu genau einem Dienst, mit Schloss-Klick. |
| 4 | **Ernstfall** | 1:15–1:32 | Blau → Rot → Blau | Dunkelrote Tinten-Ranken (Rauch, kein Glitch) kriechen auf drei Backup-Würfel zu (3-2-1). Der unveränderbare Würfel bekommt ein Schloss. Die Ranken schlagen ein, eine Brechungs-Schockwelle läuft durch den Würfel, die Ranken weichen zurück. „Zurückspielen“: **die ganze Szene spult mit Bewegungsunschärfe zurück**, und die Würfel fliegen in den Server. „Hoffnung“ steht dünn und blass da und wird durch „getestet ✓“ ersetzt. |
| 5 | **Betrieb** | 1:32–1:50 | Grün | Eine 3D-Rechnung, deren Betrag unkontrolliert hochdreht (Ziffern verwischen, keine erfundene Zahl). Das Papier faltet sich zum Papierflieger und fliegt die Umzugsbahn zum echten Proxmox-Screenshot entlang, daneben gestrichelt die „Rückweg“-Bahn. Danach ein Rolling Update: Container-Blöcke tauschen nacheinander, während die Statuslinie grün bleibt. |
| 6 | **Kein Mockup** | 1:50–2:03 | Grün | Der echte Grafana-Screenshot (3360 px) liegt wie eine Landschaft im Raum. Die Kamera fliegt tief darüber, Kippung ca. 60°, Tiefenunschärfe, und die Kurven zeichnen sich live nach. Stempel in Mono: **KEIN MOCKUP · ECHTES MONITORING**. Proxmox- und PBS-Screenshots fächern daneben auf. |
| 7 | **Ein Team** | 2:03–2:25 | Rosa | Alle Objekte der Nacht (Schaufenster, Firewall-Wand, Backup-Würfel, Dashboard) treffen sich in einer Bahn, die Kamera umkreist sie, und das Logo-Band schnürt sie zu **einem** Bündel: „Ein Team.“ „Kein Ticketstapel“: ein Turm aus Ticketzetteln (#48213 …) bricht mit Starrkörper-Physik zusammen. Dokumentationsseiten (Runbooks, Zugänge, Konfiguration) fliegen in einen Ordner, der übergeben wird. „Keine Black Box“: ein schwarzer Würfel klappt zu einem offenen Drahtgitter-Würfel auf. **Keine Stockfoto-Person** in dieser Szene. |
| 8 | **07:00** | 2:25–2:40 | Spektrum | Der Rollzähler dreht auf **07:00:00**. Morgenlicht: Der blaue Hero-Schein der Website steigt von unten auf wie ein Sonnenaufgang, Nachtblau wird Blau/Mint. Die Weckerglocke aus Szene 1 schläft ein und löst sich auf. **Das Band windet sich und faltet sich zum 3D-„N“** (Blender-Heldenshot, ein Glanzlicht wandert darüber). Die Wortmarke schreibt sich aus, dazu Claim und CTA-Pill „Erstgespräch · 30 Minuten · kostenlos“ sowie **nomissuccess.de**. Die Signalfaden-Kante am Lockup ist Marken-Moment 2. Das Lockup steht bis zum letzten Bild. |
| — | **Abspann** | +2 s | Nacht | Animierte Endkarte „made by / Anthony“, der letzte Akkord klingt darunter aus. |

**9:16:** eigene Timeline mit umgebauten Layouts statt Beschnitt. Safe Area oben 215 px, unten 405 px, seitlich 90 px.

---

## 5. Look

- **Bühne:** Nacht `#0b0c14` / `#161826` mit dem Hero-Rezept der Website: treibende Farbfelder in Rosa, Mint, Violett und Blau, die langsam wandern. Also genau der Markenschein von nomissuccess.de, nicht nachgebaut.
- **Schrift:** **Manrope** (200–800) und **JetBrains Mono** für Labels und Uhr. Beide OFL und schon im Projekt vorhanden. SF Pro nehme ich bewusst nicht, weil die Lizenz nur für Apple-Oberflächen gilt. Überschriften in Weiß, Farbe nur auf Band, Kanten und Akzenten. Satz asymmetrisch: Text links, Objekt rechts.
- **Farbregister:** Rosa/Violett = Web & Entwicklung, Blau = Netzwerk & Security, Grün = Betrieb & Monitoring, Rosa = Menschen. Voller Verlauf nur auf dem Band und den zwei Marken-Momenten.
- **Apple-Finish:** echte **Bewegungsunschärfe** (Fusion Renderer3D/Transform, Verschluss 180°, plus ResolveFX Motion Blur bei schnellen Moves), Tiefenunschärfe, Filmkorn, Glow und Light Rays dosiert, Film Look Creator als Abschluss.
- **Resolve-21-Werkzeuge, die zum Einsatz kommen:** Fusion 3D (Renderer3D, Shape3D, Text3D, Kamera und Licht), Partikel (pEmitter), die über 100 neuen Motion-Graphics-Werkzeuge (Krokodove), **Fusion-Animation, die vom Fairlight-Ton gesteuert wird** (Band und Glow pulsieren zur Musik), MultiText/Text+, Depth Map v2 für 2,5D-Parallaxe auf Fotos.

---

## 6. Ton

- **Stimme:** Google Gemini TTS über die Google-API. Der Schlüssel liegt nur lokal und kommt nie ins Repo. Eine Stimme und ein Modell für alle Zeilen, nie mischen. Regie im Stil-Feld: klar, warm, zügig, modern, wie ein Produktfilm und nicht wie ein Nachrichtensprecher.
- **Stimmauswahl:** vier Gemini-Stimmen lesen v01–v03, du wählst. Vorab-Tipp für B2B: eine klare, mittlere Männer- oder Frauenstimme Mitte 30, kein Nachrichtensprecher-Ton.
- **Musik:** eigene Musik aus Code (numpy), um die Stimme herum arrangiert. Der Bogen folgt der Nacht: Am Anfang ist das Ticken der Uhr der Hi-Hat, dann baut sich der Track auf, im Ernstfall wird er dünner, bei „Ein Team“ hebt er an, und am Morgen löst er sich warm auf. Der letzte Akkord klingt unter dem Abspann aus. **Mindestens 15 dB unter der Stimme**, solange gesprochen wird, und kein Limiter, der die Stimme quetscht.
- **SFX:** Die SSD-Bibliothek aus der Vorlage (`/Volumes/Extreme Pro/…`) ist ein Mac-Pfad und auf diesem PC nicht vorhanden. Standard ist darum eine **eigene Geräuschbibliothek aus Code** mit Whoosh, Hit, Riser, UI-Tick, Lichtschalter, Schloss, Papier, Physik-Geklapper und Logo-Sting. Jedes Wort und jeder Titel bekommt einen Whoosh, jede Landung einen Hit, jede Produkt- und Logo-Bewegung einen eigenen Ton. Alles wird platziert und per EQ aus dem Sprachband gehalten.
- **Fairlight-Spuren:** A1 Stimme · A2 Musik · A3 Whoosh · A4 Hits · A5 UI/Mechanik · A6 Riser/Übergänge · A7 Logo/Signatur · A8 Atmo, darüber SFX-Bus und Master.
- **Master:** **−14 LUFS / −1 dBTP** (Fairlight-Normalisierung, Gegenmessung mit ffmpeg ebur128) und ein Handylautsprecher-Test (RMS-Anteil über 400 Hz ≥ 45 %).

---

## 7. Assets

**Vorhanden, eigenes Material, kein Download nötig:**
- Logo als Vektor: `nomissuccess-website/marketing/build/profil/logo-vektor.svg`, dazu `logo.png`
- Schriften: `nomissuccess-werbevideo/assets/fonts/` (Manrope, JetBrains Mono, OFL)
- **Echte Screenshots:** `grafana-dashboard.png` (3360×2100), `grafana-monitoring.png`, `proxmox-ui.png`, `pbs-dashboard.png`, `opnsense-ui.png`, `nextcloud-ui.png`, `webseite-eigen.png` (nomissuccess.de selbst)
- **Pexels-Symbolbilder** (schon auf der Website, kommerziell frei): Serverraum-Gang, Racks, Glasfaser, Netzwerkkabel, Chip, Code, Entwicklung
- Echter Quellcode der Website für die Schicht-Explosion in Szene 2

**Neu, nur nach deinem OK herunterladen (Pexels):**
1. Nachtstadt / Straße bei Nacht mit Bokeh (Szene 1)
2. Dunkles Büro bei Nacht, Monitorlicht (Szene 1)
3. Morgendämmerung über Dächern (Szene 8)
4. Leeres Großraumbüro am Morgen (Szene 8, optional)

Ich zeige dir vorher Vorschaubilder und Links. Jede Quelle und Lizenz kommt in `assets_in/CREDITS.md`.

**3D (Blender 5.2, selbst gebaut):** Logo-Band mit N-Faltung, Bausteinturm, Backup-Würfel, Ticketturm, Schlüssel, Rechnung/Papierflieger.

---

## 8. Arbeitsweise in Resolve

- Projekt **„nomissuccess Spot 2026“** mit den Timelines **„nomissuccess 16x9“** und **„nomissuccess 9x16“** (kein Doppelpunkt im Namen, den lehnt die API ab), 60 fps.
- Videospuren: V1 Bühne/Fotos · V2 Blender-Renders (mit Alpha) · V3 Fusion-Szenen · V4 Typo · V5 Overlay-Ebene (Uhr, Leiste, Korn) · darüber eine Einstellungsebene für das Grading.
- **Szenenmarker** für alle 8 Szenen, die Fusion-Comps werden per Skript aus den gemessenen Sprachzeiten erzeugt. Das Projekt bleibt so reproduzierbar und trotzdem in Resolve von Hand voll editierbar.
- Kontrolle **nur über die Resolve-API** und über Einzelbilder aus dem gerenderten MP4. Keine Bildschirmaufnahmen, keine Screenshots von deinem Bildschirm.
- **Wichtig für dich:** Die Wiedergabe-Framerate lässt sich per API nicht setzen. Einmal von Hand: **Shift+9 → Haupteinstellungen → Wiedergabe-Framerate 60.** Sonst spielt die Timeline in Zeitlupe.

---

## 9. Reihenfolge

1. **Brief → dein OK** ← wir sind hier
2. **Assets:** Vorschaubilder der 4 Pexels-Kandidaten → dein OK → Download
3. **Stimmen:** 4 Gemini-Proben → du wählst
4. **Look-Test (neu):** ca. 15 s mit drei Signatur-Moves (Band-Fahrt, Schicht-Explosion der Website, N-Faltung), damit wir das Animationsniveau festlegen, bevor 2:30 gebaut werden
5. **PREVIEW v1 (16:9)** in voller Länge + Kontaktbogen → dein Feedback, so lange, bis du freigibst
6. **Finals:** 16:9 + 9:16 je mit Ton und stumm, Stems (Stimme, Musik, SFX je Kategorie), README und das editierbare Resolve-Projekt mit Spuren und Markern

---

## 10. Credits / Kosten

- **Gemini TTS:** Free Tier, also **0 €**. Engpass ist das Tageskontingent von ca. 10 Anfragen je Modell und Tag. Für heute ist es bereits verbraucht und wird um **09:00 Uhr** zurückgesetzt.
  Plan: 4 Stimmproben + Sprechertext in 3 Blöcken × 2 Takes (6 Anfragen, Aufteilung in Zeilen per lokalem Whisper) + ca. 3 Nachtakes ≈ **13 Anfragen**, also ca. zwei Tage. Notfall-Alternative ohne Kontingent: der eingebaute **Resolve-21-Sprachgenerator** (Deutsch-Qualität ungeprüft).
- Resolve, Blender, Musik und SFX laufen lokal, **0 €**. Pexels kostet nichts.

---

## 11. Offene Entscheidungen (mit meinem Vorschlag)

1. **Konzept:** A „Nachtschicht“ mit dem Logo-Band als rotem Faden?
2. **Monatsmodell (v07):** Die Website sagt „Projekt-Festpreis + monatliche Hosting-/Pflegepauschale“. Du sagst „wir bauen sie, ihr zahlt monatlich“. Was genau zahlt der Kunde monatlich? Ist der **Bau selbst** in der Monatsrate drin (also kein Einmalpreis), oder nur Hosting und Pflege? Der Satz muss stimmen. Danach muss ggf. auch die Website angepasst werden.
3. **Aussprache:** „nomissuccess“ = „No-Miss-Success“ (englisch)?
4. **Musik:** eigene Musik aus Code (Vorschlag, volle Kontrolle über jeden Treffer) oder zusätzlich ein Test mit **Lyria 3.5** über denselben Gemini-Schlüssel?
5. **Downloads:** die 4 Pexels-Kandidaten aus Abschnitt 7 suchen und vorzeigen?
6. **Abspann:** „made by / Anthony“, oder ein anderer Name?
7. **Freigabe:** Soll Simon den fertigen Spot vor der Veröffentlichung abnehmen (die Werbeaussagen laufen unter seinem Namen)?
