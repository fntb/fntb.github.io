import type { Site, Metadata, Socials } from "@types";

export const SITE: Site = {
  NAME: "@fntb",
  EMAIL: "fntb@lomail.org",
  NUM_POSTS_ON_HOMEPAGE: 3,
};

export const HOME: Metadata = {
  TITLE: "Home",
  DESCRIPTION: "Je publie sur mes apprentissages en ml. J'essaie de faire le lien entre math et implémentation.",
};

export const BLOG: Metadata = {
  TITLE: "Blog",
  DESCRIPTION: "Une collection de publications sur divers sujets de ml.",
};

export const SOCIALS: Socials = [
  { 
    NAME: "github",
    HREF: "https://github.com/fntb"
  },
];
